from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any


ORDERS_SEED: dict[str, dict[str, Any]] = {
    "O1001": {
        "order_id": "O1001",
        "user_id": "U001",
        "status": "DELIVERED",
        "paid_amount": 59.90,
        "refundable_amount": 59.90,
        "delivered_at": "2026-09-20T10:00:00+08:00",
        "items": [
            {
                "item_id": "I1001",
                "name": "陶瓷杯",
                "price": 59.90,
                "quantity": 1,
            }
        ],
    },
    "O1002": {
        "order_id": "O1002",
        "user_id": "U001",
        "status": "SHIPPED",
        "paid_amount": 129.00,
        "refundable_amount": 129.00,
        "delivered_at": None,
        "items": [
            {
                "item_id": "I1002",
                "name": "蓝牙耳机",
                "price": 129.00,
                "quantity": 1,
            }
        ],
    },
}


SHIPMENTS_SEED: dict[str, dict[str, Any]] = {
    "O1001": {
        "order_id": "O1001",
        "shipment_id": "S1001",
        "status": "DELIVERED",
        "exception_code": None,
        "latest_event": {
            "time": "2026-09-20T10:00:00+08:00",
            "description": "已签收",
        },
    },
    "O1002": {
        "order_id": "O1002",
        "shipment_id": "S1002",
        "status": "IN_TRANSIT",
        "exception_code": "NO_UPDATE_72H",
        "latest_event": {
            "time": "2026-09-19T08:00:00+08:00",
            "description": "运输中",
        },
    },
}


class InMemoryStore:
    """练习版内存业务数据仓库。

    负责保存订单、物流、工单和退款数据，并提供最小的业务校验、
    幂等控制和退款额度扣减。
    """

    def __init__(
        self,
        orders: dict[str, dict[str, Any]] | None = None,
        shipments: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        source_orders = ORDERS_SEED if orders is None else orders
        source_shipments = SHIPMENTS_SEED if shipments is None else shipments

        self._orders_seed = deepcopy(source_orders)
        self._shipments_seed = deepcopy(source_shipments)

        self.reset()

    def reset(self) -> None:
        """把所有业务数据恢复到初始状态。"""
        self._orders = deepcopy(self._orders_seed)
        self._shipments = deepcopy(self._shipments_seed)

        self._tickets_by_key: dict[str, dict[str, Any]] = {}
        self._ticket_requests: dict[str, dict[str, Any]] = {}

        self._refunds_by_key: dict[str, dict[str, Any]] = {}
        self._refund_requests: dict[str, dict[str, Any]] = {}

    def get_order(
        self,
        order_id: str,
    ) -> dict[str, Any] | None:
        """根据订单号查询订单。"""
        order = self._orders.get(order_id)

        if order is None:
            return None

        return deepcopy(order)

    def get_shipment(
        self,
        order_id: str,
    ) -> dict[str, Any] | None:
        """根据订单号查询物流。"""
        shipment = self._shipments.get(order_id)

        if shipment is None:
            return None

        return deepcopy(shipment)

    def create_ticket(
        self,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """创建售后工单。

        相同幂等键和相同请求重复调用时，返回第一次创建的工单。
        相同幂等键配合不同请求时，拒绝执行。
        """
        if not idempotency_key:
            raise ValueError("IDEMPOTENCY_KEY_REQUIRED")

        existing = self._tickets_by_key.get(idempotency_key)

        if existing is not None:
            original_request = self._ticket_requests[idempotency_key]

            if original_request != payload:
                raise ValueError("IDEMPOTENCY_KEY_CONFLICT")

            return deepcopy(existing)

        order_id = payload.get("order_id")

        if not isinstance(order_id, str):
            raise ValueError("ORDER_ID_REQUIRED")

        if order_id not in self._orders:
            raise ValueError("ORDER_NOT_FOUND")

        ticket_id = self._stable_id(
            prefix="T",
            idempotency_key=idempotency_key,
        )

        ticket = {
            **deepcopy(payload),
            "ticket_id": ticket_id,
            "status": "OPEN",
        }

        self._tickets_by_key[idempotency_key] = ticket
        self._ticket_requests[idempotency_key] = deepcopy(payload)

        return deepcopy(ticket)

    def submit_refund(
        self,
        *,
        order_id: str,
        user_id: str,
        amount: float,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """创建退款并扣减订单可退款额度。

        相同幂等键和相同请求重复调用时，返回第一次退款结果，
        不会再次扣减额度。
        """
        if not idempotency_key:
            raise ValueError("IDEMPOTENCY_KEY_REQUIRED")

        request_data = {
            "order_id": order_id,
            "user_id": user_id,
            "amount": amount,
        }

        existing = self._refunds_by_key.get(idempotency_key)

        if existing is not None:
            original_request = self._refund_requests[idempotency_key]

            if original_request != request_data:
                raise ValueError("IDEMPOTENCY_KEY_CONFLICT")

            return deepcopy(existing)

        if amount <= 0:
            raise ValueError("REFUND_AMOUNT_MUST_BE_POSITIVE")

        if amount > 10_000:
            raise ValueError("REFUND_AMOUNT_EXCEEDS_SINGLE_LIMIT")

        rounded_amount = round(amount, 2)

        if abs(amount - rounded_amount) > 1e-9:
            raise ValueError("REFUND_AMOUNT_MAX_TWO_DECIMAL_PLACES")

        order = self._orders.get(order_id)

        if order is None:
            raise ValueError("ORDER_NOT_FOUND")

        if order["user_id"] != user_id:
            raise PermissionError("ORDER_OWNER_MISMATCH")

        if rounded_amount > order["refundable_amount"]:
            raise ValueError("REFUND_AMOUNT_EXCEEDS_LIMIT")

        refund_id = self._stable_id(
            prefix="R",
            idempotency_key=idempotency_key,
        )

        refund = {
            "refund_id": refund_id,
            "order_id": order_id,
            "user_id": user_id,
            "amount": rounded_amount,
            "currency": "CNY",
            "status": "SUBMITTED",
        }

        order["refundable_amount"] = round(
            order["refundable_amount"] - rounded_amount,
            2,
        )

        self._refunds_by_key[idempotency_key] = refund
        self._refund_requests[idempotency_key] = request_data

        return deepcopy(refund)

    def get_ticket(
        self,
        ticket_id: str,
    ) -> dict[str, Any] | None:
        """根据工单号查询工单。"""
        for ticket in self._tickets_by_key.values():
            if ticket["ticket_id"] == ticket_id:
                return deepcopy(ticket)

        return None

    def list_tickets(self) -> list[dict[str, Any]]:
        """返回全部工单。"""
        return [
            deepcopy(ticket)
            for ticket in self._tickets_by_key.values()
        ]

    def list_refunds(self) -> list[dict[str, Any]]:
        """返回全部退款记录。"""
        return [
            deepcopy(refund)
            for refund in self._refunds_by_key.values()
        ]

    def counts(self) -> dict[str, int]:
        """返回工单和退款数量，主要用于测试。"""
        return {
            "tickets": len(self._tickets_by_key),
            "refunds": len(self._refunds_by_key),
        }

    @staticmethod
    def _stable_id(
        *,
        prefix: str,
        idempotency_key: str,
    ) -> str:
        """根据幂等键生成稳定的业务编号。"""
        digest = sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()

        return prefix + digest[:10].upper()


business_store = InMemoryStore()