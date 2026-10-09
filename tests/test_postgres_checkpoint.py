import os
from uuid import uuid4

import pytest

from mini_after_sales.domain import (
    ApprovalRequest,
    RunRequest,
)
from mini_after_sales.runtime import (
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
    """测试前后重置内存业务数据。"""

    business_store.reset()

    yield

    business_store.reset()


async def test_postgres_recovers_after_runtime_restart():
    """关闭第一个 Runtime 后，用新连接恢复暂停任务。

    该测试证明 Checkpoint 真正存放在 PostgreSQL，
    而不是 Python 进程中的 Runtime 对象里。
    """

    postgres_uri = os.getenv(
        "POSTGRES_TEST_URI"
    )

    if not postgres_uri:
        pytest.skip(
            "需要设置 POSTGRES_TEST_URI "
            "才能运行 PostgreSQL 集成测试"
        )

    # 每次测试使用不同 thread_id，
    # 避免与数据库中的旧测试数据冲突。
    thread_id = (
        f"postgres-{uuid4().hex}"
    )

    settings = Settings(
        checkpoint_backend="postgres",
        postgres_uri=postgres_uri,
        _env_file=None,
    )

    # 第一个 Runtime 启动退款任务并在审批节点暂停。
    async with create_runtime(
        settings
    ) as first_runtime:
        waiting = await first_runtime.run(
            RunRequest(
                thread_id=thread_id,
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

    # 上面的 async with 已经关闭第一个数据库连接。
    # 这里创建一个全新的 Runtime 和 PostgreSQL 连接。
    async with create_runtime(
        settings
    ) as second_runtime:
        saved_state = (
            await second_runtime.get_state(
                thread_id
            )
        )

        assert (
            saved_state["status"]
            == "WAITING_APPROVAL"
        )

        completed = (
            await second_runtime.approve(
                thread_id,
                ApprovalRequest(
                    approved=True,
                    reviewer_id="CS001",
                ),
            )
        )

    assert (
        completed.status
        == "COMPLETED"
    )

    assert business_store.counts() == {
        "tickets": 1,
        "refunds": 1,
    }