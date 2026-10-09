import pytest

from mini_after_sales.domain import (
    RunRequest,
)
from mini_after_sales.graph import (
    agent_graph,
    run_agent,
)
from mini_after_sales.router import (
    LOGISTICS_TOOLS,
)
from mini_after_sales.store import (
    business_store,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条测试前后恢复初始业务数据。"""

    business_store.reset()

    yield

    business_store.reset()


async def test_graph_contains_expected_nodes():
    """编译后的 Graph 应包含阶段 8 的核心节点。"""

    graph = agent_graph.get_graph()

    assert "understand" in graph.nodes
    assert "select_tools" in graph.nodes
    assert "logistics_flow" in graph.nodes
    assert "compose" in graph.nodes


async def test_graph_requests_order_id_when_missing():
    """物流请求没有订单号时不能调用业务工具。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-missing-order",
            user_id="U001",
            message="帮我查询物流",
        )
    )

    assert (
        response.status
        == "NEED_MORE_INFO"
    )
    assert "订单号" in response.message
    assert response.tool_trace == []

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_graph_handles_unknown_intent():
    """无法识别的请求应要求用户补充诉求。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-unknown",
            user_id="U001",
            message="你好",
        )
    )

    assert (
        response.status
        == "NEED_MORE_INFO"
    )
    assert (
        response.data["intent"]
        == "unknown"
    )
    assert response.tool_trace == []


async def test_graph_completes_normal_logistics_query():
    """正常物流不创建异常工单。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-normal",
            user_id="U001",
            message="查询 O1001 的物流",
        )
    )

    assert response.status == "COMPLETED"
    assert (
        response.data["intent"]
        == "logistics"
    )
    assert (
        response.data["order_id"]
        == "O1001"
    )
    assert (
        response.data["shipment"]["status"]
        == "DELIVERED"
    )
    assert (
        response.data[
            "logistics_exception"
        ]["has_exception"]
        is False
    )
    assert "ticket" not in response.data
    assert "已签收" in response.message

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_normal_logistics_tool_trace():
    """正常物流流程应调用三个只读工具。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-trace-normal",
            user_id="U001",
            message="查询 O1001 的物流",
        )
    )

    tool_names = [
        item["tool"]
        for item in response.tool_trace
    ]

    assert tool_names == [
        "get_order",
        "get_tracking_events",
        "detect_logistics_exception",
    ]


async def test_graph_creates_ticket_for_exception():
    """发现物流异常时应创建工单。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-exception",
            user_id="U001",
            message=(
                "O1002 快递三天没更新"
            ),
        )
    )

    assert response.status == "COMPLETED"

    assert (
        response.data[
            "logistics_exception"
        ]["has_exception"]
        is True
    )
    assert (
        response.data[
            "logistics_exception"
        ]["exception_code"]
        == "NO_UPDATE_72H"
    )

    ticket = response.data["ticket"]

    assert ticket["ticket_id"].startswith(
        "T"
    )
    assert ticket["order_id"] == "O1002"
    assert ticket["status"] == "OPEN"
    assert ticket["category"] == (
        "LOGISTICS_EXCEPTION"
    )

    assert ticket["ticket_id"] in (
        response.message
    )

    assert (
        business_store.counts()["tickets"]
        == 1
    )


async def test_exception_flow_tool_trace():
    """异常流程应额外调用 create_ticket。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-trace-exception",
            user_id="U001",
            message=(
                "查询 O1002 的物流，"
                "快递一直没更新"
            ),
        )
    )

    tool_names = [
        item["tool"]
        for item in response.tool_trace
    ]

    assert tool_names == [
        "get_order",
        "get_tracking_events",
        "detect_logistics_exception",
        "create_ticket",
    ]


async def test_graph_records_selected_tools():
    """Graph 应保存从 Router 筛选出的工具列表。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-tools",
            user_id="U001",
            message="查询 O1001 的物流",
        )
    )

    assert (
        response.data["selected_tools"]
        == list(LOGISTICS_TOOLS)
    )


async def test_structured_order_id_overrides_message():
    """结构化订单号的优先级高于消息中的订单号。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-override",
            user_id="U001",
            message="查询 O1002 的物流",
            order_id="O1001",
        )
    )

    assert response.status == "COMPLETED"
    assert (
        response.data["order_id"]
        == "O1001"
    )
    assert (
        response.data["shipment"]["status"]
        == "DELIVERED"
    )
    assert "ticket" not in response.data


async def test_graph_rejects_wrong_order_owner():
    """用户不能通过 Graph 查询其他用户的订单。"""

    response = await run_agent(
        RunRequest(
            thread_id="thread-wrong-owner",
            user_id="U999",
            message="查询 O1001 的物流",
        )
    )

    assert response.status == "FAILED"
    assert response.message == (
        "请求处理失败。"
    )
    assert (
        response.data["error_type"]
        == "RuntimeError"
    )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_exception_ticket_is_idempotent():
    """同一会话重复运行不能创建两张相同工单。"""

    request = RunRequest(
        thread_id="thread-idempotent",
        user_id="U001",
        message="O1002 快递三天没更新",
    )

    first = await run_agent(request)
    second = await run_agent(request)

    assert first.status == "COMPLETED"
    assert second.status == "COMPLETED"

    assert (
        first.data["ticket"]["ticket_id"]
        == second.data["ticket"]["ticket_id"]
    )

    assert (
        business_store.counts()["tickets"]
        == 1
    )