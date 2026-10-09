from __future__ import annotations

from typing import Any, Literal

from mcp.server import MCPServer
from pydantic import BaseModel, Field

mcp = MCPServer("policy-service")

class EligibilityQuery(BaseModel):
    """退款资格查询参数。"""

    order_status: str

    reason_code: Literal[
        "DAMAGED_ITEM",
        "OTHER",
    ]

    evidence_provided: bool = False


class RefundCalculation(BaseModel):
    """退款金额计算参数。"""

    paid_amount: float = Field(
        gt=0
    )

    refundable_amount: float = Field(
        ge=0
    )

    requested_amount: float | None = Field(
        default=None,
        gt=0,
    )

@mcp.tool()
def check_return_eligibility(
    query: EligibilityQuery,
) -> dict[str, Any]:
    """判断订单是否符合退款政策。"""

    if (
        query.reason_code == "DAMAGED_ITEM"
        and not query.evidence_provided
    ):
        return {
            "eligible": False,
            "reason_code": "EVIDENCE_REQUIRED",
            "policy_id": (
                "DAMAGED_ITEM_REFUND"
            ),
            "approval_required": False,
        }

    if query.order_status != "DELIVERED":
        return {
            "eligible": False,
            "reason_code": (
                "ORDER_NOT_DELIVERED"
            ),
            "policy_id": (
                "DAMAGED_ITEM_REFUND"
            ),
            "approval_required": False,
        }

    return {
        "eligible": True,
        "reason_code": (
            "DAMAGED_ITEM_ACCEPTED"
        ),
        "policy_id": (
            "DAMAGED_ITEM_REFUND"
        ),
        "approval_required": True,
    }

@mcp.tool()
def calculate_refund(
    query: RefundCalculation,
) -> dict[str, Any]:
    """计算允许退款的最大金额。"""

    ceiling = min(
        query.paid_amount,
        query.refundable_amount,
    )

    if query.requested_amount is None:
        amount = ceiling
    else:
        amount = min(
            query.requested_amount,
            ceiling,
        )

    return {
        "approved_ceiling": round(
            amount,
            2,
        ),
        "currency": "CNY",
    }