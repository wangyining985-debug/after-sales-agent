import pytest

from mini_after_sales.gateway import (
    MCPGateway,
    TOOL_REGISTRY,
    gateway,
)
from mini_after_sales.store import (
    business_store,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条测试前后重置内存业务数据。"""

    business_store.reset()

    yield

    business_store.reset()


async def test_registry_contains_all_tools():
    """网关应注册当前阶段的全部 MCP 工具。"""

    assert set(TOOL_REGISTRY) == {
        "get_order",
        "get_payment_status",
        "get_tracking_events",
        "detect_logistics_exception",
        "check_return_eligibility",
        "calculate_refund",
        "create_ticket",
        "submit_refund",
    }


async def test_registry_contains_tool_metadata():
    """工具注册表应保存服务和风险信息。"""

    get_order = TOOL_REGISTRY["get_order"]
    submit_refund = TOOL_REGISTRY[
        "submit_refund"
    ]

    assert get_order.service == "order"
    assert get_order.risk == "LOW"
    assert get_order.read_only is True

    assert submit_refund.service == "ticket"
    assert submit_refund.risk == "HIGH"
    assert submit_refund.read_only is False


async def test_gateway_calls_order_service():
    """网关应把订单工具转发给订单 MCP Server。"""

    result = await gateway.call(
        "get_order",
        {
            "query": {
                "order_id": "O1001",
                "user_id": "U001",
            }
        },
    )

    assert result["order_id"] == "O1001"
    assert result["user_id"] == "U001"
    assert result["status"] == "DELIVERED"


async def test_gateway_calls_logistics_service():
    """网关应把物流工具转发给物流 MCP Server。"""

    result = await gateway.call(
        "detect_logistics_exception",
        {
            "query": {
                "order_id": "O1002",
            }
        },
    )

    assert result == {
        "order_id": "O1002",
        "has_exception": True,
        "exception_code": "NO_UPDATE_72H",
    }


async def test_gateway_calls_policy_service():
    """网关应把政策工具转发给政策 MCP Server。"""

    result = await gateway.call(
        "check_return_eligibility",
        {
            "query": {
                "order_status": "DELIVERED",
                "reason_code": "DAMAGED_ITEM",
                "evidence_provided": True,
            }
        },
    )

    assert result["eligible"] is True
    assert result["approval_required"] is True
    assert (
        result["policy_id"]
        == "DAMAGED_ITEM_REFUND"
    )


async def test_gateway_calls_ticket_service():
    """网关应能通过工单 MCP Server 创建工单。"""

    result = await gateway.call(
        "create_ticket",
        {
            "command": {
                "order_id": "O1002",
                "user_id": "U001",
                "category": (
                    "LOGISTICS_EXCEPTION"
                ),
                "summary": (
                    "物流超过72小时未更新"
                ),
                "idempotency_key": (
                    "gateway-ticket-key-0001"
                ),
            }
        },
    )

    assert result["ticket_id"].startswith("T")
    assert result["order_id"] == "O1002"
    assert result["status"] == "OPEN"

    assert (
        business_store.counts()["tickets"]
        == 1
    )


async def test_gateway_rejects_unknown_tool():
    """未注册的工具不能通过网关调用。"""

    with pytest.raises(
        ValueError,
        match="TOOL_NOT_ALLOWED",
    ):
        await gateway.call(
            "delete_order",
            {
                "order_id": "O1001",
            },
        )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_gateway_converts_business_error():
    """MCP 业务错误应转换成统一的 RuntimeError。"""

    with pytest.raises(
        RuntimeError,
        match="MCP_TOOL_ERROR get_order",
    ):
        await gateway.call(
            "get_order",
            {
                "query": {
                    "order_id": "O1001",
                    "user_id": "U999",
                }
            },
        )


async def test_gateway_converts_schema_error():
    """MCP 参数校验错误也应转换成统一异常。"""

    with pytest.raises(
        RuntimeError,
        match="MCP_TOOL_ERROR get_order",
    ):
        await gateway.call(
            "get_order",
            {
                "query": {
                    "order_id": "bad-order-id",
                    "user_id": "U001",
                }
            },
        )


async def test_gateway_preserves_idempotency():
    """经过网关重复调用，底层幂等性仍然有效。"""

    arguments = {
        "command": {
            "order_id": "O1002",
            "user_id": "U001",
            "category": (
                "LOGISTICS_EXCEPTION"
            ),
            "summary": (
                "物流超过72小时未更新"
            ),
            "idempotency_key": (
                "gateway-ticket-key-0001"
            ),
        }
    }

    first = await gateway.call(
        "create_ticket",
        arguments,
    )

    second = await gateway.call(
        "create_ticket",
        arguments,
    )

    assert first == second

    assert (
        business_store.counts()["tickets"]
        == 1
    )


async def test_gateway_submits_approved_refund():
    """网关可以转发已经携带人工审批信息的退款。"""

    result = await gateway.call(
        "submit_refund",
        {
            "command": {
                "order_id": "O1001",
                "user_id": "U001",
                "amount": 10,
                "reason_code": "DAMAGED_ITEM",
                "idempotency_key": (
                    "gateway-refund-key-0001"
                ),
                "approval": {
                    "approved": True,
                    "reviewer_id": "CS001",
                },
            }
        },
    )

    assert result["refund_id"].startswith("R")
    assert result["order_id"] == "O1001"
    assert result["amount"] == 10
    assert result["status"] == "SUBMITTED"

    assert (
        business_store.counts()["refunds"]
        == 1
    )


async def test_gateway_instances_are_independent():
    """不同网关对象拥有独立的服务映射字典。"""

    first = MCPGateway()
    second = MCPGateway()

    assert first is not second
    assert first._servers is not second._servers
    assert set(first._servers) == {
        "order",
        "logistics",
        "policy",
        "ticket",
    }