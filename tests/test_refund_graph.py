import pytest

from mini_after_sales.domain import (
    ApprovalRequest,
    RunRequest,
)
from mini_after_sales.graph import (
    agent_graph,
    resume_agent,
    run_agent,
)
from mini_after_sales.store import (
    business_store,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条测试前后重置业务数据。

    每个测试使用不同的 thread_id，
    避免 InMemorySaver 中的 Checkpoint 相互影响。
    """

    business_store.reset()

    yield

    business_store.reset()


async def _start_refund(
    thread_id: str,
    *,
    order_id: str = "O1001",
    evidence_provided: bool = True,
):
    """构造并启动一条退款请求。"""

    return await run_agent(
        RunRequest(
            thread_id=thread_id,
            user_id="U001",
            message=(
                f"{order_id} 商品破损，"
                "我要退款"
            ),
            evidence_provided=(
                evidence_provided
            ),
        )
    )


async def test_graph_contains_refund_nodes():
    """Graph 应包含退款、审批和执行节点。"""

    graph = agent_graph.get_graph()

    assert "refund_flow" in graph.nodes
    assert "approval" in graph.nodes
    assert "execute_refund" in graph.nodes


async def test_refund_requires_evidence():
    """破损退款没有证据时不能进入人工审批。"""

    response = await _start_refund(
        "refund-missing-evidence",
        evidence_provided=False,
    )

    assert (
        response.status
        == "NEED_MORE_INFO"
    )
    assert "破损照片" in response.message

    assert (
        response.data["policy"][
            "reason_code"
        ]
        == "EVIDENCE_REQUIRED"
    )

    # 未进入金额计算和人工审批。
    tool_names = [
        trace["tool"]
        for trace in response.tool_trace
    ]

    assert tool_names == [
        "get_order",
        "get_payment_status",
        "check_return_eligibility",
    ]

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_undelivered_order_is_rejected():
    """尚未签收的订单不符合当前破损退款政策。"""

    response = await _start_refund(
        "refund-not-delivered",
        order_id="O1002",
        evidence_provided=True,
    )

    assert response.status == "REJECTED"

    assert (
        response.data["policy"][
            "reason_code"
        ]
        == "ORDER_NOT_DELIVERED"
    )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_eligible_refund_waits_for_approval():
    """符合政策的退款必须在执行写操作前暂停。"""

    response = await _start_refund(
        "refund-waiting-approval"
    )

    assert (
        response.status
        == "WAITING_APPROVAL"
    )
    assert "等待人工审批" in (
        response.message
    )

    approval_request = response.data[
        "approval_request"
    ]

    assert (
        approval_request["action"]
        == "submit_refund"
    )
    assert (
        approval_request["order_id"]
        == "O1001"
    )
    assert (
        approval_request["amount"]
        == 59.9
    )
    assert (
        approval_request["currency"]
        == "CNY"
    )

    # interrupt 之前只允许调用四个只读工具。
    tool_names = [
        trace["tool"]
        for trace in response.tool_trace
    ]

    assert tool_names == [
        "get_order",
        "get_payment_status",
        "check_return_eligibility",
        "calculate_refund",
    ]

    # 尚未审批，所以不能创建工单或退款。
    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_rejected_approval_does_not_refund():
    """人工拒绝后不能产生任何业务写入。"""

    thread_id = "refund-human-rejected"

    waiting = await _start_refund(
        thread_id
    )

    assert (
        waiting.status
        == "WAITING_APPROVAL"
    )

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=False,
            reviewer_id="CS001",
            comment="证据不充分",
        ),
    )

    assert response.status == "REJECTED"
    assert "审核未通过" in (
        response.message
    )

    assert (
        response.data[
            "rejection_reason"
        ]
        == "HUMAN_REJECTED"
    )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_approval_requires_reviewer_id():
    """批准退款时必须记录审核人。"""

    thread_id = (
        "refund-missing-reviewer"
    )

    waiting = await _start_refund(
        thread_id
    )

    assert (
        waiting.status
        == "WAITING_APPROVAL"
    )

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=True,
        ),
    )

    assert response.status == "REJECTED"

    assert (
        response.data[
            "rejection_reason"
        ]
        == "REVIEWER_ID_REQUIRED"
    )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_approved_refund_is_submitted():
    """人工批准后应创建工单并提交退款。"""

    thread_id = "refund-approved"

    waiting = await _start_refund(
        thread_id
    )

    assert (
        waiting.status
        == "WAITING_APPROVAL"
    )

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
            comment="同意退款",
        ),
    )

    assert response.status == "COMPLETED"

    ticket = response.data["ticket"]
    refund = response.data["refund"]

    assert ticket["ticket_id"].startswith(
        "T"
    )
    assert ticket["category"] == (
        "DAMAGED_ITEM_REFUND"
    )

    assert refund["refund_id"].startswith(
        "R"
    )
    assert refund["order_id"] == "O1001"
    assert refund["amount"] == 59.9
    assert refund["status"] == "SUBMITTED"

    assert business_store.counts() == {
        "tickets": 1,
        "refunds": 1,
    }

    # 完整退款后，可退款额度应变为 0。
    order = business_store.get_order(
        "O1001"
    )

    assert order is not None
    assert (
        order["refundable_amount"]
        == 0
    )


async def test_approved_refund_tool_trace():
    """完整退款的工具轨迹应包含审批前后的六个工具。"""

    thread_id = "refund-tool-trace"

    await _start_refund(thread_id)

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
        ),
    )

    tool_names = [
        trace["tool"]
        for trace in response.tool_trace
    ]

    assert tool_names == [
        "get_order",
        "get_payment_status",
        "check_return_eligibility",
        "calculate_refund",
        "create_ticket",
        "submit_refund",
    ]


async def test_reviewer_can_reduce_refund_amount():
    """审核人可以降低金额，但不能提高政策上限。"""

    thread_id = "refund-partial"

    waiting = await _start_refund(
        thread_id
    )

    assert (
        waiting.data["refund_amount"]
        == 59.9
    )

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
            approved_amount=20,
        ),
    )

    assert response.status == "COMPLETED"
    assert (
        response.data["refund"]["amount"]
        == 20
    )

    order = business_store.get_order(
        "O1001"
    )

    assert order is not None
    assert (
        order["refundable_amount"]
        == 39.9
    )


async def test_reviewer_cannot_exceed_policy_limit():
    """人工审批不能突破政策计算出的金额上限。"""

    thread_id = (
        "refund-amount-too-large"
    )

    await _start_refund(thread_id)

    response = await resume_agent(
        thread_id,
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
            approved_amount=60,
        ),
    )

    assert response.status == "REJECTED"

    assert (
        response.data[
            "rejection_reason"
        ]
        == "APPROVED_AMOUNT_INVALID"
    )

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_repeated_resume_is_idempotent():
    """同一个已完成任务重复恢复不能重复退款。"""

    thread_id = (
        "refund-repeated-resume"
    )

    await _start_refund(thread_id)

    approval = ApprovalRequest(
        approved=True,
        reviewer_id="CS001",
    )

    first = await resume_agent(
        thread_id,
        approval,
    )

    second = await resume_agent(
        thread_id,
        approval,
    )

    assert first.status == "COMPLETED"
    assert second.status == "COMPLETED"

    assert (
        first.data["refund"]["refund_id"]
        == second.data["refund"]["refund_id"]
    )

    assert (
        first.data["ticket"]["ticket_id"]
        == second.data["ticket"]["ticket_id"]
    )

    assert business_store.counts() == {
        "tickets": 1,
        "refunds": 1,
    }