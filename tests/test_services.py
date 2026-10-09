import pytest
from pydantic import ValidationError

from mini_after_sales.domain import (
    OrderQuery,
    RefundSubmission,
)
from mini_after_sales.mcp_servers.logistics import (
    ShipmentQuery,
    detect_logistics_exception,
    get_tracking_events,
)
from mini_after_sales.mcp_servers.order import (
    get_order,
    get_payment_status,
)
from mini_after_sales.mcp_servers.policy import (
    EligibilityQuery,
    RefundCalculation,
    calculate_refund,
    check_return_eligibility,
)
from mini_after_sales.mcp_servers.ticket import (
    TicketCreation,
    create_ticket,
    submit_refund,
)
from mini_after_sales.store import business_store


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条服务测试开始前重置全局业务数据。"""
    business_store.reset()

    yield

    business_store.reset()


def valid_ticket_command() -> TicketCreation:
    return TicketCreation(
        order_id="O1002",
        user_id="U001",
        category="LOGISTICS_EXCEPTION",
        summary="物流超过72小时未更新",
        idempotency_key=(
            "ticket-service-0001"
        ),
    )


def valid_refund_command() -> RefundSubmission:
    return RefundSubmission(
        order_id="O1001",
        user_id="U001",
        amount=10,
        reason_code="DAMAGED_ITEM",
        idempotency_key=(
            "refund-service-0001"
        ),
        approval={
            "approved": True,
            "reviewer_id": "CS001",
        },
    )


# ============================================================
# 订单服务
# ============================================================


def test_order_service_returns_owned_order():
    order = get_order(
        OrderQuery(
            order_id="O1001",
            user_id="U001",
        )
    )

    assert order["order_id"] == "O1001"
    assert order["user_id"] == "U001"
    assert order["status"] == "DELIVERED"


def test_order_service_rejects_unknown_order():
    with pytest.raises(
        ValueError,
        match="ORDER_NOT_FOUND",
    ):
        get_order(
            OrderQuery(
                order_id="O9999",
                user_id="U001",
            )
        )


def test_order_service_rejects_wrong_owner():
    with pytest.raises(
        PermissionError,
        match="ORDER_OWNER_MISMATCH",
    ):
        get_order(
            OrderQuery(
                order_id="O1001",
                user_id="U999",
            )
        )


def test_payment_status_returns_amounts():
    payment = get_payment_status(
        OrderQuery(
            order_id="O1001",
            user_id="U001",
        )
    )

    assert payment == {
        "order_id": "O1001",
        "payment_status": "PAID",
        "paid_amount": 59.9,
        "refundable_amount": 59.9,
    }


def test_payment_status_checks_owner():
    with pytest.raises(
        PermissionError,
        match="ORDER_OWNER_MISMATCH",
    ):
        get_payment_status(
            OrderQuery(
                order_id="O1001",
                user_id="U999",
            )
        )


# ============================================================
# 物流服务
# ============================================================


def test_tracking_service_returns_normal_shipment():
    shipment = get_tracking_events(
        ShipmentQuery(
            order_id="O1001"
        )
    )

    assert shipment["status"] == "DELIVERED"
    assert shipment["exception_code"] is None


def test_tracking_service_returns_abnormal_shipment():
    shipment = get_tracking_events(
        ShipmentQuery(
            order_id="O1002"
        )
    )

    assert shipment["status"] == "IN_TRANSIT"
    assert (
        shipment["exception_code"]
        == "NO_UPDATE_72H"
    )


def test_tracking_service_rejects_unknown_shipment():
    with pytest.raises(
        ValueError,
        match="SHIPMENT_NOT_FOUND",
    ):
        get_tracking_events(
            ShipmentQuery(
                order_id="O9999"
            )
        )


def test_shipment_query_rejects_invalid_order_id():
    with pytest.raises(ValidationError):
        ShipmentQuery(
            order_id="1001"
        )


def test_detects_normal_logistics():
    result = detect_logistics_exception(
        ShipmentQuery(
            order_id="O1001"
        )
    )

    assert result == {
        "order_id": "O1001",
        "has_exception": False,
        "exception_code": None,
    }


def test_detects_logistics_exception():
    result = detect_logistics_exception(
        ShipmentQuery(
            order_id="O1002"
        )
    )

    assert result == {
        "order_id": "O1002",
        "has_exception": True,
        "exception_code": (
            "NO_UPDATE_72H"
        ),
    }


# ============================================================
# 政策服务
# ============================================================


def test_damaged_item_requires_evidence():
    result = check_return_eligibility(
        EligibilityQuery(
            order_status="DELIVERED",
            reason_code="DAMAGED_ITEM",
            evidence_provided=False,
        )
    )

    assert result["eligible"] is False
    assert (
        result["reason_code"]
        == "EVIDENCE_REQUIRED"
    )
    assert (
        result["approval_required"]
        is False
    )


def test_undelivered_order_is_not_eligible():
    result = check_return_eligibility(
        EligibilityQuery(
            order_status="SHIPPED",
            reason_code="DAMAGED_ITEM",
            evidence_provided=True,
        )
    )

    assert result["eligible"] is False
    assert (
        result["reason_code"]
        == "ORDER_NOT_DELIVERED"
    )


def test_delivered_damaged_item_is_eligible():
    result = check_return_eligibility(
        EligibilityQuery(
            order_status="DELIVERED",
            reason_code="DAMAGED_ITEM",
            evidence_provided=True,
        )
    )

    assert result["eligible"] is True
    assert (
        result["reason_code"]
        == "DAMAGED_ITEM_ACCEPTED"
    )
    assert (
        result["approval_required"]
        is True
    )


def test_other_reason_does_not_require_evidence():
    result = check_return_eligibility(
        EligibilityQuery(
            order_status="DELIVERED",
            reason_code="OTHER",
            evidence_provided=False,
        )
    )

    assert result["eligible"] is True


def test_refund_calculation_uses_remaining_amount():
    result = calculate_refund(
        RefundCalculation(
            paid_amount=100,
            refundable_amount=80,
        )
    )

    assert result == {
        "approved_ceiling": 80,
        "currency": "CNY",
    }


def test_refund_calculation_respects_requested_amount():
    result = calculate_refund(
        RefundCalculation(
            paid_amount=100,
            refundable_amount=80,
            requested_amount=50,
        )
    )

    assert result["approved_ceiling"] == 50


def test_refund_calculation_caps_requested_amount():
    result = calculate_refund(
        RefundCalculation(
            paid_amount=100,
            refundable_amount=80,
            requested_amount=90,
        )
    )

    assert result["approved_ceiling"] == 80


def test_refund_calculation_never_exceeds_paid_amount():
    result = calculate_refund(
        RefundCalculation(
            paid_amount=50,
            refundable_amount=80,
        )
    )

    assert result["approved_ceiling"] == 50


def test_refund_calculation_validates_input():
    with pytest.raises(ValidationError):
        RefundCalculation(
            paid_amount=0,
            refundable_amount=80,
        )


# ============================================================
# 工单服务
# ============================================================


def test_ticket_service_creates_ticket():
    ticket = create_ticket(
        valid_ticket_command()
    )

    assert ticket["ticket_id"].startswith("T")
    assert ticket["order_id"] == "O1002"
    assert (
        ticket["category"]
        == "LOGISTICS_EXCEPTION"
    )
    assert ticket["status"] == "OPEN"

    assert business_store.counts() == {
        "tickets": 1,
        "refunds": 0,
    }


def test_ticket_service_is_idempotent():
    command = valid_ticket_command()

    first = create_ticket(command)
    second = create_ticket(command)

    assert first == second
    assert (
        business_store.counts()["tickets"]
        == 1
    )


def test_ticket_service_rejects_unknown_order():
    command = TicketCreation(
        order_id="O9999",
        user_id="U001",
        category="LOGISTICS_EXCEPTION",
        summary="不存在订单的工单",
        idempotency_key=(
            "ticket-service-9999"
        ),
    )

    with pytest.raises(
        ValueError,
        match="ORDER_NOT_FOUND",
    ):
        create_ticket(command)


def test_ticket_service_rejects_wrong_owner():
    command = TicketCreation(
        order_id="O1002",
        user_id="U999",
        category="LOGISTICS_EXCEPTION",
        summary="无权访问订单",
        idempotency_key=(
            "ticket-service-owner"
        ),
    )

    with pytest.raises(
        PermissionError,
        match="ORDER_OWNER_MISMATCH",
    ):
        create_ticket(command)

    assert (
        business_store.counts()["tickets"]
        == 0
    )


def test_ticket_command_validates_arguments():
    with pytest.raises(ValidationError):
        TicketCreation(
            order_id="1002",
            user_id="U001",
            category="",
            summary="",
            idempotency_key="short",
        )


# ============================================================
# 退款服务
# ============================================================


def test_refund_service_submits_approved_refund():
    refund = submit_refund(
        valid_refund_command()
    )

    assert refund["refund_id"].startswith("R")
    assert refund["order_id"] == "O1001"
    assert refund["amount"] == 10
    assert refund["status"] == "SUBMITTED"

    order = business_store.get_order(
        "O1001"
    )
    assert order is not None
    assert order["refundable_amount"] == 49.9

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 1,
    }


def test_refund_service_is_idempotent():
    command = valid_refund_command()

    first = submit_refund(command)
    second = submit_refund(command)

    assert first == second
    assert (
        business_store.counts()["refunds"]
        == 1
    )

    order = business_store.get_order(
        "O1001"
    )
    assert order is not None
    assert order["refundable_amount"] == 49.9


@pytest.mark.parametrize(
    "approval",
    [
        {
            "approved": False,
            "reviewer_id": "CS001",
        },
        {
            "approved": True,
            "reviewer_id": None,
        },
        {},
    ],
)
def test_refund_service_requires_human_approval(
    approval: dict,
):
    command = RefundSubmission(
        order_id="O1001",
        user_id="U001",
        amount=10,
        reason_code="DAMAGED_ITEM",
        idempotency_key=(
            "refund-service-0001"
        ),
        approval=approval,
    )

    with pytest.raises(
        PermissionError,
        match="HUMAN_APPROVAL_REQUIRED",
    ):
        submit_refund(command)

    assert (
        business_store.counts()["refunds"]
        == 0
    )


def test_refund_service_rejects_unknown_order():
    command = RefundSubmission(
        order_id="O9999",
        user_id="U001",
        amount=10,
        reason_code="DAMAGED_ITEM",
        idempotency_key=(
            "refund-service-9999"
        ),
        approval={
            "approved": True,
            "reviewer_id": "CS001",
        },
    )

    with pytest.raises(
        ValueError,
        match="ORDER_NOT_FOUND",
    ):
        submit_refund(command)


def test_refund_service_rejects_wrong_owner():
    command = RefundSubmission(
        order_id="O1001",
        user_id="U999",
        amount=10,
        reason_code="DAMAGED_ITEM",
        idempotency_key=(
            "refund-service-owner"
        ),
        approval={
            "approved": True,
            "reviewer_id": "CS001",
        },
    )

    with pytest.raises(
        PermissionError,
        match="ORDER_OWNER_MISMATCH",
    ):
        submit_refund(command)

    assert (
        business_store.counts()["refunds"]
        == 0
    )


def test_refund_service_rejects_amount_over_limit():
    command = RefundSubmission(
        order_id="O1001",
        user_id="U001",
        amount=60,
        reason_code="DAMAGED_ITEM",
        idempotency_key=(
            "refund-service-limit"
        ),
        approval={
            "approved": True,
            "reviewer_id": "CS001",
        },
    )

    with pytest.raises(
        ValueError,
        match="REFUND_AMOUNT_EXCEEDS_LIMIT",
    ):
        submit_refund(command)

    order = business_store.get_order(
        "O1001"
    )
    assert order is not None
    assert order["refundable_amount"] == 59.9