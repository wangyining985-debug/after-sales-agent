import pytest
from pydantic import ValidationError

from mini_after_sales.domain import (
    AgentResponse,
    ApprovalRequest,
    Intent,
    OrderQuery,
    RefundSubmission,
    RunRequest,
)


def valid_refund_payload() -> dict:
    """返回一份合法退款参数，每次调用都创建一个新字典。"""
    return {
        "order_id": "O1001",
        "user_id": "U001",
        "amount": 59.90,
        "reason_code": "DAMAGED_ITEM",
        "idempotency_key": "refund-key-00001",
        "approval": {
            "approved": True,
            "reviewer_id": "CS001",
        },
    }


# ============================================================
# Intent
# ============================================================


def test_intent_contains_expected_values():
    assert Intent.LOGISTICS.value == "logistics"
    assert Intent.REFUND.value == "refund"
    assert Intent.UNKNOWN.value == "unknown"


def test_intent_rejects_unknown_value():
    with pytest.raises(ValueError):
        Intent("exchange")


# ============================================================
# RunRequest
# ============================================================


def test_run_request_accepts_valid_minimum_data():
    request = RunRequest(
        thread_id="thread-001",
        user_id="U001",
        message="帮我查询物流",
    )

    assert request.thread_id == "thread-001"
    assert request.user_id == "U001"
    assert request.message == "帮我查询物流"
    assert request.order_id is None
    assert request.evidence_provided is False


def test_run_request_accepts_order_and_evidence():
    request = RunRequest(
        thread_id="thread-002",
        user_id="U001",
        message="O1001 商品破损，申请退款",
        order_id="O1001",
        evidence_provided=True,
    )

    assert request.order_id == "O1001"
    assert request.evidence_provided is True


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("thread_id", ""),
        ("thread_id", "t" * 129),
        ("user_id", ""),
        ("user_id", "u" * 65),
        ("message", ""),
        ("message", "m" * 2001),
        ("order_id", "O12"),
        ("order_id", "o1001"),
        ("order_id", "O" + "1" * 21),
    ],
)
def test_run_request_rejects_invalid_fields(
    field_name: str,
    invalid_value: object,
):
    payload = {
        "thread_id": "thread-001",
        "user_id": "U001",
        "message": "查询物流",
        "order_id": "O1001",
    }
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        RunRequest(**payload)


# ============================================================
# ApprovalRequest
# ============================================================


def test_approval_request_uses_defaults():
    request = ApprovalRequest(approved=True)

    assert request.approved is True
    assert request.reviewer_id is None
    assert request.comment == ""
    assert request.approved_amount is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("reviewer_id", ""),
        ("reviewer_id", "r" * 65),
        ("comment", "c" * 501),
        ("approved_amount", -0.01),
    ],
)
def test_approval_request_rejects_invalid_fields(
    field_name: str,
    invalid_value: object,
):
    payload = {
        "approved": True,
        "reviewer_id": "CS001",
        "comment": "证据完整",
        "approved_amount": 59.9,
    }
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ApprovalRequest(**payload)


def test_approval_amount_zero_is_schema_valid():
    request = ApprovalRequest(
        approved=True,
        reviewer_id="CS001",
        approved_amount=0,
    )

    assert request.approved_amount == 0


def test_approval_field_is_required():
    with pytest.raises(ValidationError):
        ApprovalRequest()


# ============================================================
# OrderQuery
# ============================================================


def test_order_query_accepts_valid_data():
    query = OrderQuery(
        order_id="O1001",
        user_id="U001",
    )

    assert query.order_id == "O1001"
    assert query.user_id == "U001"


@pytest.mark.parametrize(
    ("order_id", "user_id"),
    [
        ("O12", "U001"),
        ("o1001", "U001"),
        ("1001", "U001"),
        ("O1001", ""),
        ("O1001", "u" * 65),
    ],
)
def test_order_query_rejects_invalid_data(
    order_id: str,
    user_id: str,
):
    with pytest.raises(ValidationError):
        OrderQuery(
            order_id=order_id,
            user_id=user_id,
        )


# ============================================================
# RefundSubmission
# ============================================================


def test_refund_submission_accepts_valid_data():
    submission = RefundSubmission(
        **valid_refund_payload()
    )

    assert submission.order_id == "O1001"
    assert submission.user_id == "U001"
    assert submission.amount == 59.9
    assert submission.reason_code == "DAMAGED_ITEM"
    assert submission.approval["approved"] is True


@pytest.mark.parametrize(
    "amount",
    [
        0,
        -1,
        10_000.01,
    ],
)
def test_refund_submission_rejects_out_of_range_amount(
    amount: float,
):
    payload = valid_refund_payload()
    payload["amount"] = amount

    with pytest.raises(ValidationError):
        RefundSubmission(**payload)


def test_refund_submission_rejects_more_than_two_decimal_places():
    payload = valid_refund_payload()
    payload["amount"] = 59.999

    with pytest.raises(
        ValidationError,
        match="金额最多保留两位小数",
    ):
        RefundSubmission(**payload)


def test_refund_submission_accepts_upper_amount_boundary():
    payload = valid_refund_payload()
    payload["amount"] = 10_000

    submission = RefundSubmission(**payload)

    assert submission.amount == 10_000


@pytest.mark.parametrize(
    "reason_code",
    [
        "BROKEN",
        "REFUND",
        "",
    ],
)
def test_refund_submission_rejects_unknown_reason(
    reason_code: str,
):
    payload = valid_refund_payload()
    payload["reason_code"] = reason_code

    with pytest.raises(ValidationError):
        RefundSubmission(**payload)


@pytest.mark.parametrize(
    "idempotency_key",
    [
        "short-key",
        "k" * 129,
    ],
)
def test_refund_submission_rejects_invalid_idempotency_key(
    idempotency_key: str,
):
    payload = valid_refund_payload()
    payload["idempotency_key"] = idempotency_key

    with pytest.raises(ValidationError):
        RefundSubmission(**payload)


def test_refund_submission_requires_approval_dict():
    payload = valid_refund_payload()
    payload["approval"] = "approved"

    with pytest.raises(ValidationError):
        RefundSubmission(**payload)


def test_refund_submission_rejects_invalid_order_id():
    payload = valid_refund_payload()
    payload["order_id"] = "1001"

    with pytest.raises(ValidationError):
        RefundSubmission(**payload)


# ============================================================
# AgentResponse
# ============================================================


def test_agent_response_uses_empty_defaults():
    response = AgentResponse(
        thread_id="thread-001",
        status="COMPLETED",
        message="任务完成",
    )

    assert response.data == {}
    assert response.tool_trace == []


def test_agent_response_mutable_defaults_are_independent():
    first = AgentResponse(
        thread_id="thread-001",
        status="COMPLETED",
        message="第一个响应",
    )
    second = AgentResponse(
        thread_id="thread-002",
        status="COMPLETED",
        message="第二个响应",
    )

    first.data["refund_id"] = "R001"
    first.tool_trace.append(
        {"tool": "get_order"}
    )

    assert second.data == {}
    assert second.tool_trace == []


@pytest.mark.parametrize(
    "invalid_status",
    [
        "SUCCESS",
        "RUNNING",
        "APPROVED",
        "",
    ],
)
def test_agent_response_rejects_internal_or_unknown_status(
    invalid_status: str,
):
    with pytest.raises(ValidationError):
        AgentResponse(
            thread_id="thread-001",
            status=invalid_status,
            message="测试响应",
        )


def test_agent_response_can_be_serialized():
    response = AgentResponse(
        thread_id="thread-001",
        status="WAITING_APPROVAL",
        message="等待人工审批",
        data={
            "approval_request": {
                "order_id": "O1001",
                "amount": 59.9,
            }
        },
        tool_trace=[
            {
                "tool": "get_order",
                "arguments": {
                    "order_id": "O1001",
                },
            }
        ],
    )

    result = response.model_dump()

    assert result["thread_id"] == "thread-001"
    assert result["status"] == "WAITING_APPROVAL"
    assert result["data"]["approval_request"]["amount"] == 59.9
    assert result["tool_trace"][0]["tool"] == "get_order"