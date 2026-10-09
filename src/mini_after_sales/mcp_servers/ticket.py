from __future__ import annotations

from typing import Any

from mcp.server import MCPServer
from pydantic import BaseModel, Field

from mini_after_sales.domain import (
    RefundSubmission,
)
from mini_after_sales.store import (
    business_store,
)

mcp = MCPServer("ticket-service")

class TicketCreation(BaseModel):
    """创建工单的参数。"""

    order_id: str = Field(
        pattern=r"^O\d{3,20}$"
    )

    user_id: str = Field(
        min_length=1,
        max_length=64,
    )

    category: str = Field(
        min_length=1,
        max_length=64,
    )

    summary: str = Field(
        min_length=1,
        max_length=500,
    )

    idempotency_key: str = Field(
        min_length=16,
        max_length=128,
    )

@mcp.tool()
def create_ticket(
    command: TicketCreation,
) -> dict[str, Any]:
    """创建售后或物流工单。"""

    order = business_store.get_order(
        command.order_id
    )

    if order is None:
        raise ValueError("ORDER_NOT_FOUND")

    if order["user_id"] != command.user_id:
        raise PermissionError(
            "ORDER_OWNER_MISMATCH"
        )

    return business_store.create_ticket(
        payload={
            "order_id": command.order_id,
            "category": command.category,
            "summary": command.summary,
        },
        idempotency_key=(
            command.idempotency_key
        ),
    )

@mcp.tool()
def submit_refund(
    command: RefundSubmission,
) -> dict[str, Any]:
    """提交经过人工审批的退款。"""

    approved = command.approval.get(
        "approved"
    )
    reviewer_id = command.approval.get(
        "reviewer_id"
    )

    if not approved or not reviewer_id:
        raise PermissionError(
            "HUMAN_APPROVAL_REQUIRED"
        )

    order = business_store.get_order(
        command.order_id
    )

    if order is None:
        raise ValueError("ORDER_NOT_FOUND")

    if order["user_id"] != command.user_id:
        raise PermissionError(
            "ORDER_OWNER_MISMATCH"
        )

    return business_store.submit_refund(
        order_id=command.order_id,
        user_id=command.user_id,
        amount=command.amount,
        idempotency_key=(
            command.idempotency_key
        ),
    )