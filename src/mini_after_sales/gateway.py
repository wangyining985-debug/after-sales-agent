from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ToolDescriptor:
    """描述一个工具属于哪个服务，以及它的风险等级。"""

    service: str
    risk: str
    read_only: bool


TOOL_REGISTRY: dict[str, ToolDescriptor] = {
    "get_order": ToolDescriptor(
        service="order",
        risk="LOW",
        read_only=True,
    ),
    "get_payment_status": ToolDescriptor(
        service="order",
        risk="LOW",
        read_only=True,
    ),
    "get_tracking_events": ToolDescriptor(
        service="logistics",
        risk="LOW",
        read_only=True,
    ),
    "detect_logistics_exception": ToolDescriptor(
        service="logistics",
        risk="LOW",
        read_only=True,
    ),
    "check_return_eligibility": ToolDescriptor(
        service="policy",
        risk="LOW",
        read_only=True,
    ),
    "calculate_refund": ToolDescriptor(
        service="policy",
        risk="LOW",
        read_only=True,
    ),
    "create_ticket": ToolDescriptor(
        service="ticket",
        risk="MEDIUM",
        read_only=False,
    ),
    "submit_refund": ToolDescriptor(
        service="ticket",
        risk="HIGH",
        read_only=False,
    ),
}


class MCPGateway:
    """统一调用不同 MCP 服务的最小网关。"""

    def __init__(self) -> None:
        self._servers = {
            "order": order_mcp,
            "logistics": logistics_mcp,
            "policy": policy_mcp,
            "ticket": ticket_mcp,
        }

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """通过工具名称调用对应的 MCP Server。"""

        descriptor = TOOL_REGISTRY.get(tool_name)

        if descriptor is None:
            raise ValueError(
                f"TOOL_NOT_ALLOWED: {tool_name}"
            )

        server = self._servers[
            descriptor.service
        ]

        async with Client(server) as client:
            result = await client.call_tool(
                tool_name,
                arguments,
            )

        if result.is_error:
            message = self._extract_text(
                result.content
            )

            raise RuntimeError(
                f"MCP_TOOL_ERROR "
                f"{tool_name}: {message}"
            )

        value = result.structured_content

        if value is None:
            value = self._parse_text_result(
                tool_name=tool_name,
                content=result.content,
            )

        if not isinstance(value, dict):
            raise RuntimeError(
                f"MCP_TOOL_INVALID_RESULT: "
                f"{tool_name}"
            )

        # 某些 MCP SDK 版本可能把真正结果包装成：
        # {"result": {...}}
        if (
            set(value) == {"result"}
            and isinstance(
                value["result"],
                dict,
            )
        ):
            value = value["result"]

        return value

    @staticmethod
    def _extract_text(
        content: list[Any],
    ) -> str:
        """提取 MCP ContentBlock 中的文本。"""

        return " ".join(
            getattr(block, "text", str(block))
            for block in content
        )

    @classmethod
    def _parse_text_result(
        cls,
        *,
        tool_name: str,
        content: list[Any],
    ) -> Any:
        """structured_content 不存在时，解析文本结果。"""

        text = cls._extract_text(content)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MCP_TOOL_INVALID_RESULT: "
                f"{tool_name}"
            ) from exc


gateway = MCPGateway()