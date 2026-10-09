from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from mini_after_sales.domain import Intent


REFUND_KEYWORDS = (
    "退款",
    "退钱",
    "破损",
    "碎了",
)

LOGISTICS_KEYWORDS = (
    "物流",
    "快递",
    "包裹",
    "到哪",
    "催件",
    "催一下",
    "签收",
)


REFUND_TOOLS = (
    "get_order",
    "get_payment_status",
    "check_return_eligibility",
    "calculate_refund",
    "create_ticket",
    "submit_refund",
)

LOGISTICS_TOOLS = (
    "get_order",
    "get_tracking_events",
    "detect_logistics_exception",
    "create_ticket",
)


class RouteDecision(BaseModel):
    """Router 对用户请求的结构化判断结果。"""

    intent: Literal[
        "logistics",
        "refund",
        "unknown",
    ]

    order_id: str | None = Field(
        default=None,
        pattern=r"^O\d{3,20}$",
    )

    candidate_tools: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    reason: str = Field(
        default="",
        max_length=300,
    )


class IntentRouter:
    """使用确定性规则识别用户意图和订单号。"""

    async def route(
        self,
        message: str,
        supplied_order_id: str | None = None,
    ) -> RouteDecision:
        """分析用户消息，返回结构化路由结果。

        supplied_order_id 是调用方通过结构化字段提供的订单号。
        它的优先级高于从自然语言中提取的订单号。
        """
        order_id = (
            supplied_order_id
            if supplied_order_id is not None
            else self._extract_order_id(message)
        )

        if self._contains_keyword(
            message,
            REFUND_KEYWORDS,
        ):
            return RouteDecision(
                intent=Intent.REFUND.value,
                order_id=order_id,
                candidate_tools=list(
                    REFUND_TOOLS
                ),
                reason="refund_keyword_matched",
            )

        if self._contains_keyword(
            message,
            LOGISTICS_KEYWORDS,
        ):
            return RouteDecision(
                intent=Intent.LOGISTICS.value,
                order_id=order_id,
                candidate_tools=list(
                    LOGISTICS_TOOLS
                ),
                reason=(
                    "logistics_keyword_matched"
                ),
            )

        return RouteDecision(
            intent=Intent.UNKNOWN.value,
            order_id=order_id,
            candidate_tools=[],
            reason="no_supported_intent",
        )

    @staticmethod
    def _extract_order_id(
        message: str,
    ) -> str | None:
        """从自然语言中提取订单号。"""
        match = re.search(
            r"\bO\d{3,20}\b",
            message,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None

        return match.group(0).upper()

    @staticmethod
    def _contains_keyword(
        message: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """判断消息是否包含任意一个关键词。"""
        return any(
            keyword in message
            for keyword in keywords
        )


router = IntentRouter()