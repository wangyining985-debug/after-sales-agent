import pytest
from pydantic import ValidationError

from mini_after_sales.settings import (
    Settings,
)


def test_memory_is_default_backend():
    """未提供配置时默认使用内存 Checkpoint。"""

    settings = Settings(
        _env_file=None
    )

    assert (
        settings.checkpoint_backend
        == "memory"
    )


def test_postgres_requires_uri():
    """选择 PostgreSQL 时必须提供连接地址。"""

    with pytest.raises(
        ValidationError,
        match="POSTGRES_URI",
    ):
        Settings(
            checkpoint_backend=(
                "postgres"
            ),
            postgres_uri="",
            _env_file=None,
        )


def test_postgres_settings_are_valid():
    """提供 URI 后可以启用 PostgreSQL。"""

    settings = Settings(
        checkpoint_backend="postgres",
        postgres_uri=(
            "postgresql://"
            "agent:secret@"
            "127.0.0.1:5432/"
            "after_sales"
        ),
        _env_file=None,
    )

    assert (
        settings.checkpoint_backend
        == "postgres"
    )

    assert (
        settings.postgres_uri
        .get_secret_value()
        == (
            "postgresql://"
            "agent:secret@"
            "127.0.0.1:5432/"
            "after_sales"
        )
    )


def test_postgres_password_is_hidden():
    """配置对象的 repr 不应显示数据库密码。"""

    settings = Settings(
        checkpoint_backend="postgres",
        postgres_uri=(
            "postgresql://"
            "agent:very-secret-password@"
            "127.0.0.1:5432/"
            "after_sales"
        ),
        _env_file=None,
    )

    assert (
        "very-secret-password"
        not in repr(settings)
    )