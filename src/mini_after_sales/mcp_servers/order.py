from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from mini_after_sales.domain import OrderQuery
from mini_after_sales.store import business_store

mcp = MCPServer("order-service")

@mcp.tool()
def get_order(
    query: OrderQuery,
) -> dict[str, Any]:
    """查询订单，并校验订单是否属于当前用户。"""
    order = business_store.get_order(
        query.order_id
    )

    if order is None:
        raise ValueError("ORDER_NOT_FOUND")

    if order["user_id"] != query.user_id:
        raise PermissionError(
            "ORDER_OWNER_MISMATCH"
        )

    return order

@mcp.tool()
def get_payment_status(
    query: OrderQuery,
) -> dict[str, Any]:
    """查询订单支付状态和当前可退款金额。"""
    order = get_order(query)

    return {
        "order_id": order["order_id"],
        "payment_status": "PAID",
        "paid_amount": order["paid_amount"],
        "refundable_amount": order[
            "refundable_amount"
        ],
    }