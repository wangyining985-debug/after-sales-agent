from __future__ import annotations

import operator
from copy import deepcopy
from hashlib import sha256
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

from mini_after_sales.domain import (
    AgentResponse,
    ApprovalRequest,
    Intent,
    RunRequest,
)
from mini_after_sales.gateway import (
    MCPGateway,
    gateway,
)
from mini_after_sales.router import (
    LOGISTICS_TOOLS,
    REFUND_TOOLS,
    router,
)


class AgentState(TypedDict, total=False):
    """LangGraph 在各节点之间传递的共享状态。

    total=False 表示这些字段不是必须一次性全部提供。
    不同节点会逐步向状态中补充数据。
    """

    # ========================================================
    # 1. 用户请求数据
    # ========================================================

    # 对话线程 ID，同时作为 LangGraph Checkpoint 的 thread_id。
    thread_id: str

    # 当前发起请求的用户 ID。
    user_id: str

    # 用户输入的自然语言。
    message: str

    # 可以由请求直接提供，也可以由 Router 从 message 中提取。
    order_id: str | None

    # 用户是否已经提供商品破损证据。
    evidence_provided: bool

    # ========================================================
    # 2. Router 产生的数据
    # ========================================================

    # 当前识别到的意图：logistics、refund 或 unknown。
    intent: str

    # Router 建议使用的工具。
    candidate_tools: list[str]

    # Router 给出当前判断的原因。
    route_reason: str

    # Graph 从候选工具中再次筛选出的工具。
    selected_tools: list[str]

    # ========================================================
    # 3. MCP 工具返回的业务数据
    # ========================================================

    order: dict[str, Any]
    payment: dict[str, Any]
    shipment: dict[str, Any]
    logistics_exception: dict[str, Any]
    policy: dict[str, Any]
    ticket: dict[str, Any]
    refund: dict[str, Any]

    # 政策允许的最高退款金额，或者人工调整后的金额。
    refund_amount: float

    # 人工审批结果。
    approval: dict[str, Any]

    # 当流程被拒绝时，记录具体原因。
    rejection_reason: str

    # ========================================================
    # 4. Graph 最终结果
    # ========================================================

    # Graph 内部可能使用 RUNNING 和 APPROVED。
    # 转换成 AgentResponse 时只会输出领域模型允许的状态。
    status: str

    # 最终给用户看的文本。
    response: str

    # Annotated + operator.add 表示：
    # 不同节点返回的 tool_trace 列表会自动累加，
    # 而不是由后一个节点覆盖前一个节点。
    tool_trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


def _trace(
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """生成一条工具调用轨迹。

    审批信息可能包含审核人等内部数据，因此不把 approval
    原样写入工具轨迹。
    """

    safe_arguments = deepcopy(arguments)

    command = safe_arguments.get("command")

    if isinstance(command, dict):
        command.pop("approval", None)

    return {
        "tool": tool,
        "arguments": safe_arguments,
        "result": result,
    }


def _idempotency_key(
    state: AgentState,
    action: str,
) -> str:
    """生成稳定的幂等键。

    同一个会话、用户、订单和业务动作始终生成相同的哈希值。
    即使 Graph 被重复恢复，也不会重复创建工单或重复退款。
    """

    raw = (
        f"{state['thread_id']}:"
        f"{state['user_id']}:"
        f"{state['order_id']}:"
        f"{action}"
    )

    return sha256(
        raw.encode("utf-8")
    ).hexdigest()


def _graph_config(
    thread_id: str,
) -> dict[str, dict[str, str]]:
    """构造 LangGraph Checkpoint 配置。

    LangGraph 使用 configurable.thread_id 区分不同任务的状态。
    启动和恢复同一个任务时，必须使用完全相同的 thread_id。
    """

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


async def understand(
    state: AgentState,
) -> dict[str, Any]:
    """节点一：识别意图并确定订单号。"""

    decision = await router.route(
        message=state["message"],
        supplied_order_id=state.get(
            "order_id"
        ),
    )

    update: dict[str, Any] = {
        "intent": decision.intent,
        "order_id": decision.order_id,
        "candidate_tools": (
            decision.candidate_tools
        ),
        "route_reason": decision.reason,
    }

    # 无法识别诉求时，不调用任何业务工具。
    if (
        decision.intent
        == Intent.UNKNOWN.value
    ):
        return {
            **update,
            "status": "NEED_MORE_INFO",
        }

    # 物流和退款都必须先知道订单号。
    if decision.order_id is None:
        return {
            **update,
            "status": "NEED_MORE_INFO",
        }

    return {
        **update,
        "status": "RUNNING",
    }


def after_understand(
    state: AgentState,
) -> str:
    """根据理解结果决定是否继续选择工具。"""

    if state["status"] == "RUNNING":
        return "select_tools"

    return "compose"


async def select_tools(
    state: AgentState,
) -> dict[str, Any]:
    """节点二：根据意图筛选允许调用的工具。

    Router 负责提出候选工具；
    Graph 负责做第二次白名单过滤和必需工具检查。
    """

    if (
        state["intent"]
        == Intent.LOGISTICS.value
    ):
        allowed_tools = set(
            LOGISTICS_TOOLS
        )

        required_tools = {
            "get_order",
            "get_tracking_events",
            "detect_logistics_exception",
        }

    elif (
        state["intent"]
        == Intent.REFUND.value
    ):
        allowed_tools = set(
            REFUND_TOOLS
        )

        # 进入人工审批前，必须完成四个只读步骤。
        required_tools = {
            "get_order",
            "get_payment_status",
            "check_return_eligibility",
            "calculate_refund",
        }

    else:
        raise RuntimeError(
            "UNSUPPORTED_INTENT"
        )

    # 保留 Router 给出的顺序，但过滤掉不属于当前流程的工具。
    selected_tools = [
        tool
        for tool in state.get(
            "candidate_tools",
            [],
        )
        if tool in allowed_tools
    ]

    if not required_tools.issubset(
        selected_tools
    ):
        raise RuntimeError(
            "REQUIRED_TOOL_MISSING"
        )

    return {
        "selected_tools": selected_tools,
    }


def after_selection(
    state: AgentState,
) -> str:
    """根据意图进入物流流程或退款流程。"""

    if (
        state["intent"]
        == Intent.LOGISTICS.value
    ):
        return "logistics_flow"

    return "refund_flow"


async def _call_tool(
    client: MCPGateway,
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """调用一个经过筛选的 MCP 工具。

    这里再次检查 selected_tools，避免 Graph 节点意外调用
    Router 和工具筛选阶段没有授权的工具。
    """

    if tool_name not in state.get(
        "selected_tools",
        [],
    ):
        raise RuntimeError(
            f"TOOL_NOT_SELECTED: {tool_name}"
        )

    result = await client.call(
        tool_name,
        arguments,
    )

    trace = _trace(
        tool=tool_name,
        arguments=arguments,
        result=result,
    )

    return result, trace


async def logistics_flow(
    state: AgentState,
) -> dict[str, Any]:
    """物流流程：查询订单、查询物流、识别异常并按需建单。"""

    order_arguments = {
        "query": {
            "order_id": state["order_id"],
            "user_id": state["user_id"],
        }
    }

    order, order_trace = await _call_tool(
        gateway,
        state,
        "get_order",
        order_arguments,
    )

    shipment_arguments = {
        "query": {
            "order_id": state["order_id"],
        }
    }

    shipment, shipment_trace = (
        await _call_tool(
            gateway,
            state,
            "get_tracking_events",
            shipment_arguments,
        )
    )

    exception, exception_trace = (
        await _call_tool(
            gateway,
            state,
            "detect_logistics_exception",
            shipment_arguments,
        )
    )

    update: dict[str, Any] = {
        "order": order,
        "shipment": shipment,
        "logistics_exception": exception,
        "status": "COMPLETED",
        "tool_trace": [
            order_trace,
            shipment_trace,
            exception_trace,
        ],
    }

    # 只有检测到物流异常时才创建工单。
    if exception["has_exception"]:
        ticket_arguments = {
            "command": {
                "order_id": state[
                    "order_id"
                ],
                "user_id": state[
                    "user_id"
                ],
                "category": (
                    "LOGISTICS_EXCEPTION"
                ),
                "summary": (
                    "物流异常："
                    f"{exception['exception_code']}"
                ),
                "idempotency_key": (
                    _idempotency_key(
                        state,
                        "logistics-ticket",
                    )
                ),
            }
        }

        ticket, ticket_trace = (
            await _call_tool(
                gateway,
                state,
                "create_ticket",
                ticket_arguments,
            )
        )

        update["ticket"] = ticket
        update["tool_trace"].append(
            ticket_trace
        )

    return update


async def refund_flow(
    state: AgentState,
) -> dict[str, Any]:
    """退款审批前的只读流程。

    此节点只查询数据和计算退款金额，不创建工单，
    也不会真正提交退款。
    """

    order_arguments = {
        "query": {
            "order_id": state["order_id"],
            "user_id": state["user_id"],
        }
    }

    # 第一步：校验订单存在并属于当前用户。
    order, order_trace = await _call_tool(
        gateway,
        state,
        "get_order",
        order_arguments,
    )

    # 第二步：读取已支付金额和当前可退款金额。
    payment, payment_trace = (
        await _call_tool(
            gateway,
            state,
            "get_payment_status",
            order_arguments,
        )
    )

    policy_arguments = {
        "query": {
            "order_status": order["status"],
            "reason_code": "DAMAGED_ITEM",
            "evidence_provided": state.get(
                "evidence_provided",
                False,
            ),
        }
    }

    # 第三步：判断是否满足退款政策。
    policy, policy_trace = (
        await _call_tool(
            gateway,
            state,
            "check_return_eligibility",
            policy_arguments,
        )
    )

    update: dict[str, Any] = {
        "order": order,
        "payment": payment,
        "policy": policy,
        "tool_trace": [
            order_trace,
            payment_trace,
            policy_trace,
        ],
    }

    # 证据缺失时，提示用户补充材料。
    if (
        not policy["eligible"]
        and policy["reason_code"]
        == "EVIDENCE_REQUIRED"
    ):
        return {
            **update,
            "status": "NEED_MORE_INFO",
        }

    # 其他不符合政策的情况直接拒绝。
    if not policy["eligible"]:
        return {
            **update,
            "status": "REJECTED",
            "rejection_reason": (
                policy["reason_code"]
            ),
        }

    amount_arguments = {
        "query": {
            "paid_amount": payment[
                "paid_amount"
            ],
            "refundable_amount": payment[
                "refundable_amount"
            ],
        }
    }

    # 第四步：计算政策允许的最高退款金额。
    amount, amount_trace = (
        await _call_tool(
            gateway,
            state,
            "calculate_refund",
            amount_arguments,
        )
    )

    # 前三条工具轨迹已经保存在 update["tool_trace"] 中。
    # 这里应该追加金额计算轨迹，而不是用新列表覆盖原来的轨迹。
    update["tool_trace"].append(
        amount_trace
    )

    update["refund_amount"] = amount[
        "approved_ceiling"
    ]

    update["status"] = (
        "WAITING_APPROVAL"
    )

    return update


def after_refund_flow(
    state: AgentState,
) -> str:
    """符合政策时进入人工审批，否则直接组织回复。"""

    if (
        state["status"]
        == "WAITING_APPROVAL"
    ):
        return "approval"

    return "compose"


def approval(
    state: AgentState,
) -> dict[str, Any]:
    """人工审批节点。

    interrupt() 会暂停 Graph，并把审批信息返回给调用方。
    恢复时，Command(resume=...) 中的数据会成为 decision。
    """

    decision = interrupt(
        {
            "action": "submit_refund",
            "order_id": state["order_id"],
            "amount": state["refund_amount"],
            "currency": "CNY",
            "policy": state["policy"],
            "question": "是否批准该退款？",
        }
    )

    # 防止调用方绕过 ApprovalRequest，直接传入错误类型。
    if not isinstance(decision, dict):
        return {
            "approval": {},
            "status": "REJECTED",
            "rejection_reason": (
                "INVALID_APPROVAL_DATA"
            ),
        }

    # 审核人明确拒绝时，不执行任何写操作。
    if not decision.get("approved"):
        return {
            "approval": decision,
            "status": "REJECTED",
            "rejection_reason": (
                "HUMAN_REJECTED"
            ),
        }

    # 批准操作必须能够追溯到具体审核人。
    if not decision.get("reviewer_id"):
        return {
            "approval": decision,
            "status": "REJECTED",
            "rejection_reason": (
                "REVIEWER_ID_REQUIRED"
            ),
        }

    approved_amount = decision.get(
        "approved_amount"
    )

    # 审核人可以降低退款金额，但不能超过政策上限。
    if approved_amount is not None:
        approved_amount = round(
            float(approved_amount),
            2,
        )

        if (
            approved_amount <= 0
            or approved_amount
            > state["refund_amount"]
        ):
            return {
                "approval": decision,
                "status": "REJECTED",
                "rejection_reason": (
                    "APPROVED_AMOUNT_INVALID"
                ),
            }

        final_amount = approved_amount

    else:
        final_amount = state[
            "refund_amount"
        ]

    return {
        "approval": decision,
        "refund_amount": final_amount,
        "status": "APPROVED",
    }


def after_approval(
    state: AgentState,
) -> str:
    """批准后执行退款，拒绝后直接组织回复。"""

    if state["status"] == "APPROVED":
        return "execute_refund"

    return "compose"


async def execute_refund(
    state: AgentState,
) -> dict[str, Any]:
    """执行经过人工审批的写操作。

    只有 approval 节点返回 APPROVED 后才能进入这里。
    工单和退款分别使用独立幂等键。
    """

    ticket_arguments = {
        "command": {
            "order_id": state["order_id"],
            "user_id": state["user_id"],
            "category": (
                "DAMAGED_ITEM_REFUND"
            ),
            "summary": "商品破损退款",
            "idempotency_key": (
                _idempotency_key(
                    state,
                    "refund-ticket",
                )
            ),
        }
    }

    ticket, ticket_trace = (
        await _call_tool(
            gateway,
            state,
            "create_ticket",
            ticket_arguments,
        )
    )

    refund_arguments = {
        "command": {
            "order_id": state["order_id"],
            "user_id": state["user_id"],
            "amount": state[
                "refund_amount"
            ],
            "reason_code": "DAMAGED_ITEM",
            "idempotency_key": (
                _idempotency_key(
                    state,
                    "refund",
                )
            ),
            "approval": state["approval"],
        }
    }

    refund, refund_trace = (
        await _call_tool(
            gateway,
            state,
            "submit_refund",
            refund_arguments,
        )
    )

    return {
        "ticket": ticket,
        "refund": refund,
        "status": "COMPLETED",
        "tool_trace": [
            ticket_trace,
            refund_trace,
        ],
    }


async def compose(
    state: AgentState,
) -> dict[str, Any]:
    """把内部状态转换成用户可以理解的文字。"""

    if state["status"] == "NEED_MORE_INFO":
        if not state.get("order_id"):
            message = (
                "请提供需要处理的订单号"
                "（例如 O1001）。"
            )

        elif (
            state.get("policy", {}).get(
                "reason_code"
            )
            == "EVIDENCE_REQUIRED"
        ):
            message = (
                "请先提供商品破损照片，"
                "再继续申请退款。"
            )

        else:
            message = (
                "目前支持物流查询和商品破损退款，"
                "请补充具体诉求。"
            )

        return {
            "response": message,
        }

    if state["status"] == "REJECTED":
        reason = state.get(
            "rejection_reason"
        )

        if reason == "HUMAN_REJECTED":
            message = (
                "人工审核未通过，"
                "本次没有执行退款。"
            )

        elif reason == "REVIEWER_ID_REQUIRED":
            message = (
                "审批人身份缺失，"
                "本次没有执行退款。"
            )

        elif (
            reason
            == "APPROVED_AMOUNT_INVALID"
        ):
            message = (
                "人工批准的退款金额不合法，"
                "本次没有执行退款。"
            )

        elif reason == "ORDER_NOT_DELIVERED":
            message = (
                "订单尚未完成签收，"
                "当前不符合退款政策。"
            )

        else:
            message = (
                "当前退款申请未通过，"
                "本次没有执行退款。"
            )

        return {
            "response": message,
        }

    if (
        state["intent"]
        == Intent.LOGISTICS.value
    ):
        ticket = state.get("ticket")

        if ticket is not None:
            return {
                "response": (
                    "检测到物流异常，"
                    f"已创建工单 "
                    f"{ticket['ticket_id']}。"
                )
            }

        shipment = state["shipment"]

        return {
            "response": (
                "当前物流状态："
                f"{shipment['status']}，"
                f"{shipment['latest_event']['description']}。"
            )
        }

    # 能执行到这里，说明退款已通过审批并提交。
    refund = state["refund"]
    ticket = state["ticket"]

    return {
        "response": (
            "退款已提交，"
            f"退款单 {refund['refund_id']}，"
            f"金额 ¥{refund['amount']:.2f}，"
            f"售后工单 {ticket['ticket_id']}。"
        )
    }


def build_graph(
    checkpointer: Any | None = None,
):
    """创建并编译售后 Agent Graph。

    checkpointer 由外部传入，方便阶段 10 将 InMemorySaver
    替换为 PostgreSQL Checkpoint。
    """

    builder = StateGraph(AgentState)

    builder.add_node(
        "understand",
        understand,
    )
    builder.add_node(
        "select_tools",
        select_tools,
    )
    builder.add_node(
        "logistics_flow",
        logistics_flow,
    )
    builder.add_node(
        "refund_flow",
        refund_flow,
    )
    builder.add_node(
        "approval",
        approval,
    )
    builder.add_node(
        "execute_refund",
        execute_refund,
    )
    builder.add_node(
        "compose",
        compose,
    )

    builder.add_edge(
        START,
        "understand",
    )

    builder.add_conditional_edges(
        "understand",
        after_understand,
        {
            "select_tools": "select_tools",
            "compose": "compose",
        },
    )

    builder.add_conditional_edges(
        "select_tools",
        after_selection,
        {
            "logistics_flow": (
                "logistics_flow"
            ),
            "refund_flow": "refund_flow",
        },
    )

    builder.add_edge(
        "logistics_flow",
        "compose",
    )

    builder.add_conditional_edges(
        "refund_flow",
        after_refund_flow,
        {
            "approval": "approval",
            "compose": "compose",
        },
    )

    builder.add_conditional_edges(
        "approval",
        after_approval,
        {
            "execute_refund": (
                "execute_refund"
            ),
            "compose": "compose",
        },
    )

    builder.add_edge(
        "execute_refund",
        "compose",
    )

    builder.add_edge(
        "compose",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer
    )


# 阶段 9 使用内存 Checkpointer。
# 它可以支持当前进程内的 interrupt 和 resume。
# 进程关闭后其中的数据会消失。
memory_checkpointer = InMemorySaver()

agent_graph = build_graph(
    checkpointer=memory_checkpointer
)


def _public_data(
    state: dict[str, Any],
) -> dict[str, Any]:
    """从内部 Graph 状态中选择允许返回给调用方的数据。"""

    public_fields = (
        "intent",
        "order_id",
        "route_reason",
        "selected_tools",
        "order",
        "payment",
        "shipment",
        "logistics_exception",
        "policy",
        "refund_amount",
        "approval",
        "ticket",
        "refund",
        "rejection_reason",
    )

    return {
        field: state[field]
        for field in public_fields
        if field in state
    }


def _state_to_response(
    thread_id: str,
    state: dict[str, Any],
) -> AgentResponse:
    """把 Graph 返回状态转换为 AgentResponse。"""

    data = _public_data(state)

    # interrupt() 发生后，LangGraph 会在结果中增加
    # __interrupt__。其中的 value 就是提供给审核人的信息。
    interrupts = state.get(
        "__interrupt__",
        [],
    )

    if interrupts:
        approval_request = getattr(
            interrupts[0],
            "value",
            {},
        )

        data["approval_request"] = (
            approval_request
        )

        return AgentResponse(
            thread_id=thread_id,
            status="WAITING_APPROVAL",
            message="退款申请等待人工审批。",
            data=data,
            tool_trace=state.get(
                "tool_trace",
                [],
            ),
        )

    return AgentResponse(
        thread_id=thread_id,
        status=state["status"],
        message=state["response"],
        data=data,
        tool_trace=state.get(
            "tool_trace",
            [],
        ),
    )


def _failed_response(
    thread_id: str,
    exc: Exception,
) -> AgentResponse:
    """把未处理异常转换成统一失败响应。"""

    return AgentResponse(
        thread_id=thread_id,
        status="FAILED",
        message="请求处理失败。",
        data={
            "error_type": (
                type(exc).__name__
            ),
        },
        tool_trace=[],
    )


async def run_agent(
    request: RunRequest,
) -> AgentResponse:
    """启动一次新的 Agent 处理流程。

    如果退款流程触发 interrupt，本函数会返回
    WAITING_APPROVAL，而不是继续执行写操作。
    """

    config = _graph_config(
        request.thread_id
    )

    try:
        state = await agent_graph.ainvoke(
            request.model_dump(),
            config=config,
        )
    except Exception as exc:
        return _failed_response(
            request.thread_id,
            exc,
        )

    return _state_to_response(
        request.thread_id,
        state,
    )


async def resume_agent(
    thread_id: str,
    approval_request: ApprovalRequest,
) -> AgentResponse:
    """使用人工审批结果恢复之前暂停的 Graph。

    thread_id 必须与启动退款流程时使用的 thread_id 相同，
    否则 LangGraph 无法找到对应的内存 Checkpoint。
    """

    config = _graph_config(thread_id)

    # exclude_none=True 可以避免把没有填写的 approved_amount
    # 以 None 的形式传入审批节点。
    resume_data = (
        approval_request.model_dump(
            exclude_none=True
        )
    )

    try:
        state = await agent_graph.ainvoke(
            Command(
                resume=resume_data
            ),
            config=config,
        )
    except Exception as exc:
        return _failed_response(
            thread_id,
            exc,
        )

    return _state_to_response(
        thread_id,
        state,
    )