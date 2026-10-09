import pytest
from pydantic import ValidationError

from mini_after_sales.router import (
    LOGISTICS_TOOLS,
    REFUND_TOOLS,
    IntentRouter,
    RouteDecision,
    router,
)


# ============================================================
# 物流意图
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_order_id"),
    [
        (
            "查询订单 O1001 的物流",
            "O1001",
        ),
        (
            "O1002 快递三天没更新",
            "O1002",
        ),
        (
            "帮我看一下包裹 O1001",
            "O1001",
        ),
        (
            "帮我催一下 O1002",
            "O1002",
        ),
        (
            "O1001 是否已经签收",
            "O1001",
        ),
    ],
)
async def test_router_recognizes_logistics_intent(
    message: str,
    expected_order_id: str,
):
    decision = await router.route(
        message
    )

    assert decision.intent == "logistics"
    assert (
        decision.order_id
        == expected_order_id
    )
    assert (
        decision.candidate_tools
        == list(LOGISTICS_TOOLS)
    )
    assert (
        decision.reason
        == "logistics_keyword_matched"
    )


# ============================================================
# 退款意图
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_order_id"),
    [
        (
            "O1001 杯子碎了，我要退款",
            "O1001",
        ),
        (
            "订单 O1001 商品破损",
            "O1001",
        ),
        (
            "O1001 帮我退钱",
            "O1001",
        ),
        (
            "申请订单 O1001 退款",
            "O1001",
        ),
    ],
)
async def test_router_recognizes_refund_intent(
    message: str,
    expected_order_id: str,
):
    decision = await router.route(
        message
    )

    assert decision.intent == "refund"
    assert (
        decision.order_id
        == expected_order_id
    )
    assert (
        decision.candidate_tools
        == list(REFUND_TOOLS)
    )
    assert (
        decision.reason
        == "refund_keyword_matched"
    )


# ============================================================
# 未知意图
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "你好",
        "今天天气怎么样",
        "帮我处理一下",
        "我想咨询一个问题",
    ],
)
async def test_router_returns_unknown_for_unsupported_intent(
    message: str,
):
    decision = await router.route(
        message
    )

    assert decision.intent == "unknown"
    assert decision.order_id is None
    assert decision.candidate_tools == []
    assert (
        decision.reason
        == "no_supported_intent"
    )


@pytest.mark.asyncio
async def test_unknown_intent_can_still_extract_order_id():
    decision = await router.route(
        "订单 O1001 怎么样了"
    )

    assert decision.intent == "unknown"
    assert decision.order_id == "O1001"
    assert decision.candidate_tools == []


# ============================================================
# 订单号提取
# ============================================================


@pytest.mark.asyncio
async def test_router_extracts_uppercase_order_id():
    decision = await router.route(
        "查询 O1001 的物流"
    )

    assert decision.order_id == "O1001"


@pytest.mark.asyncio
async def test_router_normalizes_lowercase_order_id_from_message():
    decision = await router.route(
        "查询 o1001 的物流"
    )

    assert decision.order_id == "O1001"


@pytest.mark.asyncio
async def test_router_returns_none_when_order_id_is_missing():
    decision = await router.route(
        "帮我查物流"
    )

    assert decision.intent == "logistics"
    assert decision.order_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "查询 O12 的物流",
        "查询 X1001 的物流",
        (
            "查询订单 "
            "O123456789012345678901"
            " 的物流"
        ),
    ],
)
async def test_router_does_not_extract_invalid_order_id(
    message: str,
):
    decision = await router.route(
        message
    )

    assert decision.intent == "logistics"
    assert decision.order_id is None


# ============================================================
# 结构化订单号优先级
# ============================================================


@pytest.mark.asyncio
async def test_supplied_order_id_is_used_when_message_has_none():
    decision = await router.route(
        message="帮我查物流",
        supplied_order_id="O1001",
    )

    assert decision.intent == "logistics"
    assert decision.order_id == "O1001"


@pytest.mark.asyncio
async def test_supplied_order_id_overrides_message_order_id():
    decision = await router.route(
        message="查询 O1002 的物流",
        supplied_order_id="O1001",
    )

    assert decision.order_id == "O1001"


@pytest.mark.asyncio
async def test_invalid_supplied_order_id_is_rejected():
    with pytest.raises(
        ValidationError
    ):
        await router.route(
            message="查询物流",
            supplied_order_id="bad-order",
        )


# ============================================================
# 意图优先级
# ============================================================


@pytest.mark.asyncio
async def test_refund_has_priority_over_logistics():
    decision = await router.route(
        "O1001 物流太慢了，我要退款"
    )

    assert decision.intent == "refund"
    assert (
        decision.candidate_tools
        == list(REFUND_TOOLS)
    )


@pytest.mark.asyncio
async def test_damage_keyword_routes_to_refund():
    decision = await router.route(
        "O1001 快递到了，但是商品破损"
    )

    assert decision.intent == "refund"


# ============================================================
# RouteDecision模型
# ============================================================


def test_route_decision_rejects_unknown_intent_value():
    with pytest.raises(
        ValidationError
    ):
        RouteDecision(
            intent="exchange",
            order_id="O1001",
            candidate_tools=[],
            reason="测试",
        )


def test_route_decision_rejects_invalid_order_id():
    with pytest.raises(
        ValidationError
    ):
        RouteDecision(
            intent="logistics",
            order_id="1001",
            candidate_tools=[],
            reason="测试",
        )


def test_route_decision_rejects_too_many_tools():
    with pytest.raises(
        ValidationError
    ):
        RouteDecision(
            intent="logistics",
            order_id="O1001",
            candidate_tools=[
                f"tool-{index}"
                for index in range(9)
            ],
            reason="测试",
        )


def test_route_decision_uses_independent_tool_lists():
    first = RouteDecision(
        intent="unknown"
    )
    second = RouteDecision(
        intent="unknown"
    )

    first.candidate_tools.append(
        "get_order"
    )

    assert second.candidate_tools == []


# ============================================================
# Router实例
# ============================================================


@pytest.mark.asyncio
async def test_separate_router_instances_are_independent():
    first_router = IntentRouter()
    second_router = IntentRouter()

    first = await first_router.route(
        "查询 O1001 的物流"
    )
    second = await second_router.route(
        "O1001 商品破损退款"
    )

    assert first.intent == "logistics"
    assert second.intent == "refund"