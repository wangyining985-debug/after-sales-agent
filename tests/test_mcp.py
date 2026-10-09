import json

import pytest
from mcp import Client

from mini_after_sales.mcp_servers.logistics import (
    mcp as logistics_mcp,
)
from mini_after_sales.mcp_servers.order import (
    mcp as order_mcp,
)
from mini_after_sales.mcp_servers.policy import (
    mcp as policy_mcp,
)
from mini_after_sales.mcp_servers.ticket import (
    mcp as ticket_mcp,
)
from mini_after_sales.store import business_store


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条MCP测试开始前重置业务数据。"""
    business_store.reset()

    yield

    business_store.reset()


def result_as_dict(result) -> dict:
    """把MCP工具返回的文本内容解析成字典。"""
    assert not result.is_error

    # 把 result.content 里每个 block 的 text 属性取出来，然后全部拼接成一个字符串
    text = "".join(
        getattr(block, "text", "")
        for block in result.content
    )

    return json.loads(text)


# ============================================================
# Order MCP
# ============================================================


async def test_order_mcp_get_order():
    async with Client(order_mcp) as client:
        result = await client.call_tool(
            "get_order",
            {
                "query": {
                    "order_id": "O1001",
                    "user_id": "U001",
                }
            },
        )

    order = result_as_dict(result)

    assert order["order_id"] == "O1001"
    assert order["user_id"] == "U001"
    assert order["status"] == "DELIVERED"


async def test_order_mcp_get_payment_status():
    async with Client(order_mcp) as client:
        result = await client.call_tool(
            "get_payment_status",
            {
                "query": {
                    "order_id": "O1001",
                    "user_id": "U001",
                }
            },
        )

    payment = result_as_dict(result)

    assert payment == {
        "order_id": "O1001",
        "payment_status": "PAID",
        "paid_amount": 59.9,
        "refundable_amount": 59.9,
    }


async def test_order_mcp_rejects_wrong_owner():
    async with Client(order_mcp) as client:
        result = await client.call_tool(
            "get_order",
            {
                "query": {
                    "order_id": "O1001",
                    "user_id": "U999",
                }
            },
        )

    assert result.is_error


async def test_order_mcp_rejects_invalid_schema():
    async with Client(order_mcp) as client:
        result = await client.call_tool(
            "get_order",
            {
                "query": {
                    "order_id": "bad-id",
                    "user_id": "U001",
                }
            },
        )

    assert result.is_error


async def test_order_mcp_rejects_missing_query_wrapper():
    async with Client(order_mcp) as client:
        result = await client.call_tool(
            "get_order",
            {
                "order_id": "O1001",
                "user_id": "U001",
            },
        )

    assert result.is_error


# ============================================================
# Logistics MCP
# ============================================================


async def test_logistics_mcp_get_tracking_events():
    async with Client(
        logistics_mcp
    ) as client:
        result = await client.call_tool(
            "get_tracking_events",
            {
                "query": {
                    "order_id": "O1002",
                }
            },
        )

    shipment = result_as_dict(result)

    assert shipment["order_id"] == "O1002"
    assert shipment["status"] == "IN_TRANSIT"
    assert (
        shipment["exception_code"]
        == "NO_UPDATE_72H"
    )


async def test_logistics_mcp_detects_exception():
    async with Client(
        logistics_mcp
    ) as client:
        result = await client.call_tool(
            "detect_logistics_exception",
            {
                "query": {
                    "order_id": "O1002",
                }
            },
        )

    exception = result_as_dict(result)

    assert exception == {
        "order_id": "O1002",
        "has_exception": True,
        "exception_code": (
            "NO_UPDATE_72H"
        ),
    }


async def test_logistics_mcp_detects_normal_delivery():
    async with Client(
        logistics_mcp
    ) as client:
        result = await client.call_tool(
            "detect_logistics_exception",
            {
                "query": {
                    "order_id": "O1001",
                }
            },
        )

    exception = result_as_dict(result)

    assert exception == {
        "order_id": "O1001",
        "has_exception": False,
        "exception_code": None,
    }


async def test_logistics_mcp_rejects_unknown_shipment():
    async with Client(
        logistics_mcp
    ) as client:
        result = await client.call_tool(
            "get_tracking_events",
            {
                "query": {
                    "order_id": "O9999",
                }
            },
        )

    assert result.is_error


# ============================================================
# Policy MCP
# ============================================================


async def test_policy_mcp_requires_evidence():
    async with Client(policy_mcp) as client:
        result = await client.call_tool(
            "check_return_eligibility",
            {
                "query": {
                    "order_status": "DELIVERED",
                    "reason_code": (
                        "DAMAGED_ITEM"
                    ),
                    "evidence_provided": False,
                }
            },
        )

    policy = result_as_dict(result)

    assert policy["eligible"] is False
    assert (
        policy["reason_code"]
        == "EVIDENCE_REQUIRED"
    )


async def test_policy_mcp_accepts_valid_refund():
    async with Client(policy_mcp) as client:
        result = await client.call_tool(
            "check_return_eligibility",
            {
                "query": {
                    "order_status": "DELIVERED",
                    "reason_code": (
                        "DAMAGED_ITEM"
                    ),
                    "evidence_provided": True,
                }
            },
        )

    policy = result_as_dict(result)

    assert policy["eligible"] is True
    assert (
        policy["approval_required"]
        is True
    )


async def test_policy_mcp_calculates_refund():
    async with Client(policy_mcp) as client:
        result = await client.call_tool(
            "calculate_refund",
            {
                "query": {
                    "paid_amount": 100,
                    "refundable_amount": 80,
                    "requested_amount": 50,
                }
            },
        )

    amount = result_as_dict(result)

    assert amount == {
        "approved_ceiling": 50,
        "currency": "CNY",
    }


async def test_policy_mcp_rejects_invalid_amount():
    async with Client(policy_mcp) as client:
        result = await client.call_tool(
            "calculate_refund",
            {
                "query": {
                    "paid_amount": 0,
                    "refundable_amount": 80,
                }
            },
        )

    assert result.is_error


# ============================================================
# Ticket MCP
# ============================================================


async def test_ticket_mcp_creates_ticket():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
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
                        "ticket-mcp-key-0001"
                    ),
                }
            },
        )

    ticket = result_as_dict(result)

    assert ticket["ticket_id"].startswith("T")
    assert ticket["order_id"] == "O1002"
    assert ticket["status"] == "OPEN"
    assert (
        business_store.counts()["tickets"]
        == 1
    )


async def test_ticket_mcp_is_idempotent_across_calls():
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
                "ticket-mcp-key-0001"
            ),
        }
    }

    async with Client(ticket_mcp) as client:
        first_result = await client.call_tool(
            "create_ticket",
            arguments,
        )

    async with Client(ticket_mcp) as client:
        second_result = await client.call_tool(
            "create_ticket",
            arguments,
        )

    first = result_as_dict(first_result)
    second = result_as_dict(second_result)

    assert first == second
    assert (
        business_store.counts()["tickets"]
        == 1
    )


async def test_ticket_mcp_rejects_wrong_owner():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
            "create_ticket",
            {
                "command": {
                    "order_id": "O1002",
                    "user_id": "U999",
                    "category": (
                        "LOGISTICS_EXCEPTION"
                    ),
                    "summary": "无权创建工单",
                    "idempotency_key": (
                        "ticket-mcp-owner-01"
                    ),
                }
            },
        )

    assert result.is_error
    assert (
        business_store.counts()["tickets"]
        == 0
    )


# ============================================================
# Refund MCP
# ============================================================


async def test_refund_mcp_requires_approval():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
            "submit_refund",
            {
                "command": {
                    "order_id": "O1001",
                    "user_id": "U001",
                    "amount": 10,
                    "reason_code": (
                        "DAMAGED_ITEM"
                    ),
                    "idempotency_key": (
                        "refund-mcp-key-0001"
                    ),
                    "approval": {
                        "approved": False,
                        "reviewer_id": "CS001",
                    },
                }
            },
        )

    assert result.is_error
    assert (
        business_store.counts()["refunds"]
        == 0
    )


async def test_refund_mcp_requires_reviewer():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
            "submit_refund",
            {
                "command": {
                    "order_id": "O1001",
                    "user_id": "U001",
                    "amount": 10,
                    "reason_code": (
                        "DAMAGED_ITEM"
                    ),
                    "idempotency_key": (
                        "refund-mcp-key-0001"
                    ),
                    "approval": {
                        "approved": True,
                    },
                }
            },
        )

    assert result.is_error
    assert (
        business_store.counts()["refunds"]
        == 0
    )


async def test_refund_mcp_submits_refund():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
            "submit_refund",
            {
                "command": {
                    "order_id": "O1001",
                    "user_id": "U001",
                    "amount": 10,
                    "reason_code": (
                        "DAMAGED_ITEM"
                    ),
                    "idempotency_key": (
                        "refund-mcp-key-0001"
                    ),
                    "approval": {
                        "approved": True,
                        "reviewer_id": "CS001",
                    },
                }
            },
        )

    refund = result_as_dict(result)

    assert refund["refund_id"].startswith("R")
    assert refund["order_id"] == "O1001"
    assert refund["amount"] == 10
    assert refund["status"] == "SUBMITTED"

    order = business_store.get_order(
        "O1001"
    )
    assert order is not None
    assert order["refundable_amount"] == 49.9

    assert (
        business_store.counts()["refunds"]
        == 1
    )


async def test_refund_mcp_is_idempotent_across_sessions():
    arguments = {
        "command": {
            "order_id": "O1001",
            "user_id": "U001",
            "amount": 10,
            "reason_code": "DAMAGED_ITEM",
            "idempotency_key": (
                "refund-mcp-key-0001"
            ),
            "approval": {
                "approved": True,
                "reviewer_id": "CS001",
            },
        }
    }

    async with Client(ticket_mcp) as client:
        first_result = await client.call_tool(
            "submit_refund",
            arguments,
        )

    async with Client(ticket_mcp) as client:
        second_result = await client.call_tool(
            "submit_refund",
            arguments,
        )

    first = result_as_dict(first_result)
    second = result_as_dict(second_result)

    assert first == second
    assert (
        business_store.counts()["refunds"]
        == 1
    )

    order = business_store.get_order(
        "O1001"
    )
    assert order is not None

    # 两个独立MCP会话只扣减一次退款额度。
    assert order["refundable_amount"] == 49.9


async def test_refund_mcp_rejects_invalid_schema():
    async with Client(ticket_mcp) as client:
        result = await client.call_tool(
            "submit_refund",
            {
                "command": {
                    "order_id": "bad-order",
                    "user_id": "U001",
                    "amount": -1,
                    "reason_code": "UNKNOWN",
                    "idempotency_key": "short",
                    "approval": {},
                }
            },
        )

    assert result.is_error
    assert (
        business_store.counts()["refunds"]
        == 0
    )