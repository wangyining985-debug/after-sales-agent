from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import (
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """从环境变量和 .env 文件读取项目配置。

    BaseSettings 会按照以下顺序读取配置：

    1. 创建 Settings 时直接传入的参数；
    2. 系统环境变量；
    3. .env 文件；
    4. 类中定义的默认值。

    环境变量不区分大小写，因此：
    checkpoint_backend 和 CHECKPOINT_BACKEND 都能被识别。
    """

    model_config = SettingsConfigDict(
        # 本地开发时自动读取项目根目录下的 .env。
        env_file=".env",

        # 明确指定 .env 使用 UTF-8。
        env_file_encoding="utf-8",

        # .env 中出现当前阶段尚未使用的字段时不报错。
        extra="ignore",

        # 环境变量名称不区分大小写。
        case_sensitive=False,
    )

    # memory：
    #   使用 InMemorySaver。
    #   适合单元测试，但进程退出后状态消失。
    #
    # postgres：
    #   使用 AsyncPostgresSaver。
    #   适合完整测试和服务运行。
    checkpoint_backend: Literal[
        "memory",
        "postgres",
    ] = "memory"

    # 使用 SecretStr，避免日志、repr 或调试信息直接打印密码。
    #
    # 默认留空，只有选择 postgres 后才要求必须配置。
    postgres_uri: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def validate_checkpoint_config(
        self,
    ) -> "Settings":
        """校验 Checkpoint 配置的组合是否合法。"""

        if (
            self.checkpoint_backend
            == "postgres"
            and not self.postgres_uri
            .get_secret_value()
            .strip()
        ):
            raise ValueError(
                "CHECKPOINT_BACKEND=postgres 时，"
                "必须配置 POSTGRES_URI"
            )

        return self


@lru_cache
def get_settings() -> Settings:
    """返回进程级配置单例。

    lru_cache 可以避免每次请求都重新读取 .env。
    测试环境如果修改了环境变量，应调用：

        get_settings.cache_clear()
    """

    return Settings()