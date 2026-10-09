from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mini_after_sales.store import business_store


class ShipmentQuery(BaseModel):
    """物流查询参数。"""

    order_id: str = Field(
        pattern=r"^O\d{3,20}$"
    )


def get_tracking_events(
    query: ShipmentQuery,
) -> dict[str, Any]:
    """查询订单的物流状态和最新轨迹。"""
    shipment = business_store.get_shipment(
        query.order_id
    )

    if shipment is None:
        raise ValueError("SHIPMENT_NOT_FOUND")

    return shipment


def detect_logistics_exception(
    query: ShipmentQuery,
) -> dict[str, Any]:
    """判断物流是否存在异常。"""
    shipment = get_tracking_events(query)

    exception_code = shipment[
        "exception_code"
    ]

    return {
        "order_id": query.order_id,
        "has_exception": (
            exception_code is not None
        ),
        "exception_code": exception_code,
    }