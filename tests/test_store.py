import pytest

from mini_after_sales.store import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    """每条测试使用一个全新的内存业务仓库。"""
    return InMemoryStore()


# ============================================================
# 订单查询
# ============================================================


def test_get_order_returns_known_order(
    store: InMemoryStore,
):
    order = store.get_order("O1001")

    assert order is not None
    assert order["order_id"] == "O1001"
    assert order["user_id"] == "U001"
    assert order["status"] == "DELIVERED"
    assert order["paid_amount"] == 59.9
    assert order["refundable_amount"] == 59.9


def test_get_order_returns_none_for_unknown_order(
    store: InMemoryStore,
):
    assert store.get_order("O9999") is None


def test_get_order_returns_a_copy(
    store: InMemoryStore,
):
    first = store.get_order("O1001")
    assert first is not None

    first["status"] = "MODIFIED"
    first["items"][0]["name"] = "被外部修改"

    second = store.get_order("O1001")
    assert second is not None

    assert second["status"] == "DELIVERED"
    assert second["items"][0]["name"] == "陶瓷杯"


# ============================================================
# 物流查询
# ============================================================


def test_get_normal_shipment(
    store: InMemoryStore,
):
    shipment = store.get_shipment("O1001")

    assert shipment is not None
    assert shipment["status"] == "DELIVERED"
    assert shipment["exception_code"] is None
    assert shipment["latest_event"]["description"] == "已签收"


def test_get_abnormal_shipment(
    store: InMemoryStore,
):
    shipment = store.get_shipment("O1002")

    assert shipment is not None
    assert shipment["status"] == "IN_TRANSIT"
    assert shipment["exception_code"] == "NO_UPDATE_72H"


def test_get_shipment_returns_none_for_unknown_order(
    store: InMemoryStore,
):
    assert store.get_shipment("O9999") is None


def test_get_shipment_returns_a_copy(
    store: InMemoryStore,
):
    first = store.get_shipment("O1002")
    assert first is not None

    first["status"] = "MODIFIED"
    first["latest_event"]["description"] = "被外部修改"

    second = store.get_shipment("O1002")
    assert second is not None

    assert second["status"] == "IN_TRANSIT"
    assert second["latest_event"]["description"] == "运输中"


# ============================================================
# 创建工单
# ============================================================


def test_create_ticket(
    store: InMemoryStore,
):
    ticket = store.create_ticket(
        {
            "order_id": "O1002",
            "category": "LOGISTICS_EXCEPTION",
            "summary": "物流超过72小时未更新",
        },
        idempotency_key="ticket-key-00001",
    )

    assert ticket["ticket_id"].startswith("T")
    assert len(ticket["ticket_id"]) == 11
    assert ticket["order_id"] == "O1002"
    assert ticket["category"] == "LOGISTICS_EXCEPTION"
    assert ticket["status"] == "OPEN"

    assert store.counts() == {
        "tickets": 1,
        "refunds": 0,
    }


def test_create_ticket_is_idempotent(
    store: InMemoryStore,
):
    payload = {
        "order_id": "O1002",
        "category": "LOGISTICS_EXCEPTION",
        "summary": "物流超过72小时未更新",
    }

    first = store.create_ticket(
        payload,
        idempotency_key="ticket-key-00001",
    )
    second = store.create_ticket(
        payload,
        idempotency_key="ticket-key-00001",
    )

    assert first == second
    assert store.counts()["tickets"] == 1


def test_ticket_id_is_stable_for_same_idempotency_key(
    store: InMemoryStore,
):
    payload = {
        "order_id": "O1002",
        "category": "LOGISTICS_EXCEPTION",
        "summary": "物流异常",
    }

    first = store.create_ticket(
        payload,
        idempotency_key="ticket-key-00001",
    )

    store.reset()

    second = store.create_ticket(
        payload,
        idempotency_key="ticket-key-00001",
    )

    assert first["ticket_id"] == second["ticket_id"]


def test_create_ticket_rejects_same_key_with_different_request(
    store: InMemoryStore,
):
    store.create_ticket(
        {
            "order_id": "O1002",
            "category": "LOGISTICS_EXCEPTION",
            "summary": "物流异常",
        },
        idempotency_key="ticket-key-00001",
    )

    with pytest.raises(
        ValueError,
        match="IDEMPOTENCY_KEY_CONFLICT",
    ):
        store.create_ticket(
            {
                "order_id": "O1001",
                "category": "REFUND",
                "summary": "另一个请求",
            },
            idempotency_key="ticket-key-00001",
        )

    assert store.counts()["tickets"] == 1


def test_create_ticket_requires_idempotency_key(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="IDEMPOTENCY_KEY_REQUIRED",
    ):
        store.create_ticket(
            {
                "order_id": "O1002",
                "category": "LOGISTICS_EXCEPTION",
                "summary": "物流异常",
            },
            idempotency_key="",
        )


def test_create_ticket_rejects_unknown_order(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="ORDER_NOT_FOUND",
    ):
        store.create_ticket(
            {
                "order_id": "O9999",
                "category": "LOGISTICS_EXCEPTION",
                "summary": "不存在的订单",
            },
            idempotency_key="ticket-key-99999",
        )


def test_get_ticket(
    store: InMemoryStore,
):
    created = store.create_ticket(
        {
            "order_id": "O1002",
            "category": "LOGISTICS_EXCEPTION",
            "summary": "物流异常",
        },
        idempotency_key="ticket-key-00001",
    )

    result = store.get_ticket(created["ticket_id"])

    assert result == created


def test_get_ticket_returns_none_when_not_found(
    store: InMemoryStore,
):
    assert store.get_ticket("T0000000000") is None


# ============================================================
# 提交退款
# ============================================================


def test_submit_refund(
    store: InMemoryStore,
):
    refund = store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    assert refund["refund_id"].startswith("R")
    assert len(refund["refund_id"]) == 11
    assert refund["order_id"] == "O1001"
    assert refund["amount"] == 10
    assert refund["currency"] == "CNY"
    assert refund["status"] == "SUBMITTED"

    order = store.get_order("O1001")
    assert order is not None
    assert order["refundable_amount"] == 49.9

    assert store.counts() == {
        "tickets": 0,
        "refunds": 1,
    }


def test_submit_refund_is_idempotent(
    store: InMemoryStore,
):
    first = store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )
    second = store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    assert first == second
    assert store.counts()["refunds"] == 1

    order = store.get_order("O1001")
    assert order is not None

    # 相同请求重复执行，额度只能扣减一次。
    assert order["refundable_amount"] == 49.9


def test_refund_id_is_stable_for_same_idempotency_key(
    store: InMemoryStore,
):
    first = store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    store.reset()

    second = store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    assert first["refund_id"] == second["refund_id"]


def test_submit_refund_rejects_same_key_with_different_request(
    store: InMemoryStore,
):
    store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    with pytest.raises(
        ValueError,
        match="IDEMPOTENCY_KEY_CONFLICT",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=20,
            idempotency_key="refund-key-00001",
        )

    order = store.get_order("O1001")
    assert order is not None

    assert order["refundable_amount"] == 49.9
    assert store.counts()["refunds"] == 1


@pytest.mark.parametrize(
    "amount",
    [
        0,
        -1,
    ],
)
def test_submit_refund_requires_positive_amount(
    store: InMemoryStore,
    amount: float,
):
    with pytest.raises(
        ValueError,
        match="REFUND_AMOUNT_MUST_BE_POSITIVE",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=amount,
            idempotency_key=f"refund-key-{amount}",
        )

    assert store.counts()["refunds"] == 0


def test_submit_refund_rejects_more_than_two_decimals(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="REFUND_AMOUNT_MAX_TWO_DECIMAL_PLACES",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=10.999,
            idempotency_key="refund-key-decimal",
        )

    assert store.counts()["refunds"] == 0


def test_submit_refund_rejects_single_limit(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="REFUND_AMOUNT_EXCEEDS_SINGLE_LIMIT",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=10_000.01,
            idempotency_key="refund-key-big-limit",
        )


def test_submit_refund_rejects_unknown_order(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="ORDER_NOT_FOUND",
    ):
        store.submit_refund(
            order_id="O9999",
            user_id="U001",
            amount=10,
            idempotency_key="refund-key-unknown",
        )


def test_submit_refund_rejects_wrong_owner(
    store: InMemoryStore,
):
    with pytest.raises(
        PermissionError,
        match="ORDER_OWNER_MISMATCH",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U999",
            amount=10,
            idempotency_key="refund-key-owner",
        )

    order = store.get_order("O1001")
    assert order is not None

    assert order["refundable_amount"] == 59.9
    assert store.counts()["refunds"] == 0


def test_submit_refund_rejects_amount_above_refundable_amount(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="REFUND_AMOUNT_EXCEEDS_LIMIT",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=60,
            idempotency_key="refund-key-limit",
        )

    order = store.get_order("O1001")
    assert order is not None

    assert order["refundable_amount"] == 59.9
    assert store.counts()["refunds"] == 0


def test_multiple_refunds_reduce_remaining_amount(
    store: InMemoryStore,
):
    store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )
    store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=20,
        idempotency_key="refund-key-00002",
    )

    order = store.get_order("O1001")
    assert order is not None

    assert order["refundable_amount"] == 29.9
    assert store.counts()["refunds"] == 2


def test_submit_refund_requires_idempotency_key(
    store: InMemoryStore,
):
    with pytest.raises(
        ValueError,
        match="IDEMPOTENCY_KEY_REQUIRED",
    ):
        store.submit_refund(
            order_id="O1001",
            user_id="U001",
            amount=10,
            idempotency_key="",
        )


# ============================================================
# 重置与测试隔离
# ============================================================


def test_reset_restores_initial_state(
    store: InMemoryStore,
):
    store.create_ticket(
        {
            "order_id": "O1002",
            "category": "LOGISTICS_EXCEPTION",
            "summary": "物流异常",
        },
        idempotency_key="ticket-key-00001",
    )

    store.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    assert store.counts() == {
        "tickets": 1,
        "refunds": 1,
    }

    store.reset()

    assert store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }

    order = store.get_order("O1001")
    assert order is not None
    assert order["refundable_amount"] == 59.9


def test_store_instances_are_independent():
    first = InMemoryStore()
    second = InMemoryStore()

    first.submit_refund(
        order_id="O1001",
        user_id="U001",
        amount=10,
        idempotency_key="refund-key-00001",
    )

    first_order = first.get_order("O1001")
    second_order = second.get_order("O1001")

    assert first_order is not None
    assert second_order is not None

    assert first_order["refundable_amount"] == 49.9
    assert second_order["refundable_amount"] == 59.9