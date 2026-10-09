from __future__ import annotations

from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from langgraph.types import Command

from mini_after_sales.domain import (
    AgentResponse,
    ApprovalRequest,
    RunRequest,
)
from mini_after_sales.graph import (
    build_graph,
)
from mini_after_sales.settings import (
    Settings,
    get_settings,
)


class AgentRuntime:
    """负责运行、暂停和恢复售后 Agent。

    Graph 只描述节点和边；
    Runtime 负责给 Graph 提供 Checkpointer，并调用 Graph。
    """

    def __init__(
        self,
        checkpointer: Any,
    ) -> None:
        """使用指定的 Checkpointer 编译 Graph。

        checkpointer 可以是：

        - InMemorySaver
        - AsyncPostgresSaver
        - 其他符合 LangGraph 接口的 Checkpointer
        """

        self.checkpointer = checkpointer

        self.graph = build_graph(
            checkpointer=checkpointer
        )

    @staticmethod
    def _config(
        thread_id: str,
    ) -> dict[str, dict[str, str]]:
        """生成 LangGraph 调用配置。

        thread_id 是查找 Checkpoint 的关键字段。

        启动和恢复同一任务时必须使用相同的 thread_id。
        """

        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    @staticmethod
    def _public_data(
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """筛选可以返回给调用方的业务数据。

        LangGraph 状态中可能包含内部执行信息，
        因此不直接把整个 result 返回。
        """

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
            field: result[field]
            for field in public_fields
            if field in result
        }

    @classmethod
    def _response(
        cls,
        thread_id: str,
        result: dict[str, Any],
    ) -> AgentResponse:
        """把 LangGraph 结果转换成 AgentResponse。"""

        data = cls._public_data(result)

        # interrupt() 暂停时，LangGraph 会额外返回
        # __interrupt__ 字段。
        interrupts = result.get(
            "__interrupt__",
            [],
        )

        if interrupts:
            # Interrupt 对象的 value 是 approval 节点
            # 传给人工审核系统的数据。
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
                message=(
                    "退款申请等待人工审批。"
                ),
                data=data,
                tool_trace=result.get(
                    "tool_trace",
                    [],
                ),
            )

        internal_status = result.get(
            "status",
            "FAILED",
        )

        # AgentState 内部还可能出现 RUNNING、APPROVED。
        # 这些中间状态不能直接暴露为 AgentResponse。
        allowed_statuses = {
            "COMPLETED",
            "NEED_MORE_INFO",
            "REJECTED",
        }

        if (
            internal_status
            not in allowed_statuses
        ):
            public_status = "FAILED"
        else:
            public_status = internal_status

        return AgentResponse(
            thread_id=thread_id,
            status=public_status,
            message=result.get(
                "response",
                "任务执行失败。",
            ),
            data=data,
            tool_trace=result.get(
                "tool_trace",
                [],
            ),
        )

    @staticmethod
    def _failed_response(
        thread_id: str,
        exc: Exception,
    ) -> AgentResponse:
        """把运行时异常转换成统一失败响应。

        当前学习阶段返回 error_type 方便调试，
        但不向调用方暴露数据库密码或完整异常堆栈。
        """

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

    async def run(
        self,
        request: RunRequest,
    ) -> AgentResponse:
        """启动一个 Agent 流程。

        如果流程进入 interrupt，本方法返回
        WAITING_APPROVAL，此时 Checkpoint 已经由
        Checkpointer 保存。
        """

        config = self._config(
            request.thread_id
        )

        try:
            result = await self.graph.ainvoke(
                request.model_dump(),
                config=config,
            )
        except Exception as exc:
            return self._failed_response(
                request.thread_id,
                exc,
            )

        return self._response(
            request.thread_id,
            result,
        )

    async def approve(
        self,
        thread_id: str,
        request: ApprovalRequest,
    ) -> AgentResponse:
        """根据人工审批结果恢复 Graph。

        Command(resume=...) 不会从 START 重新运行，
        而是读取 Checkpoint，从 interrupt 所在节点继续。
        """

        config = self._config(thread_id)

        resume_data = request.model_dump(
            exclude_none=True
        )

        try:
            result = await self.graph.ainvoke(
                Command(
                    resume=resume_data
                ),
                config=config,
            )
        except Exception as exc:
            return self._failed_response(
                thread_id,
                exc,
            )

        return self._response(
            thread_id,
            result,
        )

    async def get_state(
        self,
        thread_id: str,
    ) -> dict[str, Any]:
        """读取指定任务的最新 Graph 状态。

        graph.aget_state() 返回的是 StateSnapshot，
        而不是普通字典。

        StateSnapshot.values 才是节点保存的状态数据。
        Runtime 在这里将它复制成普通 dict，
        让调用方可以使用 state["status"] 访问。
        """

        snapshot = await self.graph.aget_state(
            self._config(thread_id)
        )

        return dict(snapshot.values)


@asynccontextmanager
async def create_runtime(
    settings: Settings | None = None,
) -> AsyncIterator[AgentRuntime]:
    """根据配置创建一个带生命周期的 Runtime。

    必须使用：

        async with create_runtime() as runtime:
            ...

    原因是 PostgreSQL Checkpointer 持有数据库连接，
    使用结束后必须正确关闭。
    """

    current_settings = (
        settings
        if settings is not None
        else get_settings()
    )

    if (
        current_settings.checkpoint_backend
        == "memory"
    ):
        # 内存模式不持有外部连接。
        yield AgentRuntime(
            InMemorySaver()
        )
        return

    postgres_uri = (
        current_settings
        .postgres_uri
        .get_secret_value()
    )

    # from_conn_string() 返回异步上下文管理器。
    # 退出 async with 时，数据库连接会自动关闭。
    async with (
        AsyncPostgresSaver
        .from_conn_string(
            postgres_uri
        )
    ) as checkpointer:
        # setup() 会创建或升级 LangGraph 所需的表。
        # 该操作是幂等的，可以在应用启动时调用。
        await checkpointer.setup()

        yield AgentRuntime(
            checkpointer
        )


# ============================================================
# 测试和学习时使用的默认内存 Runtime
# ============================================================
#
# API 阶段不要直接使用这个全局对象。
# API 应在 lifespan 中调用 create_runtime()，
# 根据环境变量选择 memory 或 postgres。

memory_runtime = AgentRuntime(
    InMemorySaver()
)

# 为了让阶段 8、9 的测试只需少量修改，
# 暂时保留 agent_graph、run_agent 和 resume_agent。
agent_graph = memory_runtime.graph


async def run_agent(
    request: RunRequest,
) -> AgentResponse:
    """使用默认内存 Runtime 启动任务。"""

    return await memory_runtime.run(
        request
    )


async def resume_agent(
    thread_id: str,
    approval_request: ApprovalRequest,
) -> AgentResponse:
    """使用默认内存 Runtime 恢复任务。"""

    return await memory_runtime.approve(
        thread_id,
        approval_request,
    )