import pytest
from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from mini_after_sales.domain import (
    ApprovalRequest,
    RunRequest,
)
from mini_after_sales.runtime import (
    AgentRuntime,
    create_runtime,
)
from mini_after_sales.settings import (
    Settings,
)
from mini_after_sales.store import (
    business_store,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def reset_business_data():
    """每条测试前后恢复业务数据。"""

    business_store.reset()

    yield

    business_store.reset()


async def test_runtime_pauses_and_resumes():
    """Runtime 应支持退款的暂停与恢复。"""

    runtime = AgentRuntime(
        InMemorySaver()
    )

    waiting = await runtime.run(
        RunRequest(
            thread_id="runtime-refund",
            user_id="U001",
            message=(
                "O1001 商品破损退款"
            ),
            evidence_provided=True,
        )
    )

    assert (
        waiting.status
        == "WAITING_APPROVAL"
    )

    # 审批前不允许产生写操作。
    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }

    completed = await runtime.approve(
        "runtime-refund",
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
        ),
    )

    assert completed.status == "COMPLETED"

    assert business_store.counts() == {
        "tickets": 1,
        "refunds": 1,
    }


async def test_checkpoint_is_owned_by_saver():
    """更换 Runtime 对象后，只要 Saver 相同就能恢复。

    这说明状态保存在 Checkpointer 中，
    而不是保存在 AgentRuntime 实例自身。
    """

    saver = InMemorySaver()

    first_runtime = AgentRuntime(saver)

    waiting = await first_runtime.run(
        RunRequest(
            thread_id=(
                "runtime-recreated"
            ),
            user_id="U001",
            message=(
                "O1001 商品破损退款"
            ),
            evidence_provided=True,
        )
    )

    assert (
        waiting.status
        == "WAITING_APPROVAL"
    )

    # 创建新的 Runtime，但继续使用同一个 Saver。
    second_runtime = AgentRuntime(saver)

    completed = await (
        second_runtime.approve(
            "runtime-recreated",
            ApprovalRequest(
                approved=True,
                reviewer_id="CS002",
            ),
        )
    )

    assert completed.status == "COMPLETED"


async def test_runtime_can_read_saved_state():
    """Runtime 应能读取 Checkpointer 中的当前状态。"""

    runtime = AgentRuntime(
        InMemorySaver()
    )

    await runtime.run(
        RunRequest(
            thread_id="runtime-state",
            user_id="U001",
            message=(
                "O1001 商品破损退款"
            ),
            evidence_provided=True,
        )
    )

    state = await runtime.get_state(
        "runtime-state"
    )

    assert state["intent"] == "refund"
    assert state["order_id"] == "O1001"
    assert (
        state["status"]
        == "WAITING_APPROVAL"
    )
    assert state["refund_amount"] == 59.9


async def test_wrong_thread_cannot_resume():
    """不同 thread_id 不能恢复另一个任务。"""

    runtime = AgentRuntime(
        InMemorySaver()
    )

    await runtime.run(
        RunRequest(
            thread_id="correct-thread",
            user_id="U001",
            message=(
                "O1001 商品破损退款"
            ),
            evidence_provided=True,
        )
    )

    response = await runtime.approve(
        "wrong-thread",
        ApprovalRequest(
            approved=True,
            reviewer_id="CS001",
        ),
    )

    assert response.status == "FAILED"

    assert business_store.counts() == {
        "tickets": 0,
        "refunds": 0,
    }


async def test_create_memory_runtime():
    """create_runtime 应支持 memory 配置。"""

    settings = Settings(
        checkpoint_backend="memory",
        _env_file=None,
    )

    async with create_runtime(
        settings
    ) as runtime:
        response = await runtime.run(
            RunRequest(
                thread_id=(
                    "factory-memory"
                ),
                user_id="U001",
                message=(
                    "查询 O1001 的物流"
                ),
            )
        )

    assert response.status == "COMPLETED"