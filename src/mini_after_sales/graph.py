from __future__ import annotations

import operator
from hashlib import sha256
from typing import Annotated, Any, TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from mini_after_sales.domain import (
    AgentResponse,
    Intent,
    RunRequest,
)
from mini_after_sales.gateway import (
    MCPGateway,
    gateway,
)
from mini_after_sales.router import (
    IntentRouter,
    router,
)


class AgentState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态。"""

    # 用户请求
    thread_id: str
    user_id: str
    message: str
    order_id: str | None
    evidence_provided: bool

    # Router 产生的数据
    intent: str
    candidate_tools: list[str]
    route_reason: str

    # Graph 筛选后的工具
    selected_tools: list[str]

    # MCP 工具返回的数据
    order: dict[str, Any]
    shipment: dict[str, Any]
    logistics_exception: dict[str, Any]
    ticket: dict[str, Any]

    # 最终处理结果
    status: str
    response: str

    # operator.add 表示不同节点返回的列表需要累加
    tool_trace: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]


def _trace(
    tool: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """生成一条工具调用轨迹。"""

    return {
        "tool": tool,
        "arguments": arguments,
        "result": result,
    }


def _idempotency_key(
    state: AgentState,
    action: str,
) -> str:
    """根据会话、用户、订单和操作生成稳定的幂等键。"""

    raw = (
        f"{state['thread_id']}:"
        f"{state['user_id']}:"
        f"{state['order_id']}:"
        f"{action}"
    )

    return sha256(
        raw.encode("utf-8")
    ).hexdigest()


async def understand(
    state: AgentState,
) -> dict[str, Any]:
    """理解用户意图并提取订单号。"""

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

    if (
        decision.intent
        == Intent.UNKNOWN.value
    ):
        return {
            **update,
            "status": "NEED_MORE_INFO",
        }

    if decision.order_id is None:
        return {
            **update,
            "status": "NEED_MORE_INFO",
        }

    # 阶段 8 只实现物流闭环。
    # 退款会在加入人工审批的阶段继续实现。
    if (
        decision.intent
        == Intent.REFUND.value
    ):
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
    """决定理解请求后进入哪个节点。"""

    if state["status"] == "RUNNING":
        return "select_tools"

    return "compose"


async def select_tools(
    state: AgentState,
) -> dict[str, Any]:
    """从 Router 候选工具中筛选物流流程允许的工具。"""

    allowed_tools = {
        "get_order",
        "get_tracking_events",
        "detect_logistics_exception",
        "create_ticket",
    }

    selected_tools = [
        tool
        for tool in state.get(
            "candidate_tools",
            [],
        )
        if tool in allowed_tools
    ]

    required_tools = {
        "get_order",
        "get_tracking_events",
        "detect_logistics_exception",
    }

    if not required_tools.issubset(
        selected_tools
    ):
        raise RuntimeError(
            "REQUIRED_TOOL_MISSING"
        )

    return {
        "selected_tools": selected_tools,
    }


async def _call_tool(
    client: MCPGateway,
    state: AgentState,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """调用经过 Graph 筛选的 MCP 工具。"""

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
    """执行完整的物流查询和异常建单流程。"""

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


async def compose(
    state: AgentState,
) -> dict[str, Any]:
    """根据 Graph 状态生成最终回复。"""

    if state["status"] == "NEED_MORE_INFO":
        if not state.get("order_id"):
            message = (
                "请提供需要处理的订单号"
                "（例如 O1001）。"
            )

        elif (
            state.get("intent")
            == Intent.REFUND.value
        ):
            message = (
                "已识别退款诉求，"
                "退款审批流程将在下一阶段接入。"
            )

        else:
            message = (
                "目前支持物流查询，"
                "请说明需要查询物流或补充具体诉求。"
            )

        return {
            "response": message,
        }

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


def build_graph():
    """构建并编译阶段 8 的 LangGraph。"""

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

    builder.add_edge(
        "select_tools",
        "logistics_flow",
    )

    builder.add_edge(
        "logistics_flow",
        "compose",
    )

    builder.add_edge(
        "compose",
        END,
    )

    return builder.compile()


agent_graph = build_graph()


async def run_agent(
    request: RunRequest,
) -> AgentResponse:
    """运行 Graph，并转换为统一的响应模型。"""

    try:
        state = await agent_graph.ainvoke(
            request.model_dump()
        )
    except Exception as exc:
        return AgentResponse(
            thread_id=request.thread_id,
            status="FAILED",
            message="请求处理失败。",
            data={
                "error_type": (
                    type(exc).__name__
                ),
            },
            tool_trace=[],
        )

    data_fields = (
        "intent",
        "order_id",
        "route_reason",
        "selected_tools",
        "order",
        "shipment",
        "logistics_exception",
        "ticket",
    )

    data = {
        field: state[field]
        for field in data_fields
        if field in state
    }

    return AgentResponse(
        thread_id=request.thread_id,
        status=state["status"],
        message=state["response"],
        data=data,
        tool_trace=state.get(
            "tool_trace",
            [],
        ),
    )