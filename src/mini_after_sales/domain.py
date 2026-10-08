from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Intent(StrEnum): # 定义用户意图
    LOGISTICS = "logistics" # 查看物流
    REFUND = "refund"  # 寻求退款
    UNKNOWN = "unknown" # 未知

class RunRequest(BaseModel): # agent请求信息的数据规范
    thread_id: str= Field(min_length=1, max_length=128) #当前对话id
    user_id: str= Field(min_length=1, max_length=64) #用户id
    message: str = Field(min_length=1, max_length=2000)#用户输入的自然语言
    order_id: str | None = Field(default=None, pattern=r"^O\d{3,20}$") #订单号(0开头的3～20位的整数)，可以暂时没有
    evidence_provided: bool = False # 用户是否提交售后证据，默认未提交

class ApprovalRequest(BaseModel): #人工审核结果的数据规范
    approved: bool # 是否赞同用户退款
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=64) # 审批id
    comment: str = Field(default="", max_length=500) # 评语
    approved_amount: float | None = Field(default=None, ge=0) # 批准的退款金额

class AgentResponse(BaseModel): # agent回复的数据规范
    thread_id: str # 当前对话id
    status: Literal[ # 当前对话状态
        "COMPLETED",
        "WAITING_APPROVAL",
        "NEED_MORE_INFO",
        "REJECTED",
        "FAILED",
    ]
    message: str 
    # 如果创建对象时没有传 data，就自动调用 dict() 创建一个新的空字典。
    data: dict[str,Any] = Field(default_factory=dict)
    # 如果创建对象时没有传 tool_trace，就自动调用 list() 创建一个新的空列表。
    tool_trace: list[dict[str,Any]] = Field(default_factory=list)

class OrderQuery(BaseModel):
    order_id: str= Field(pattern=r"^O\d{3,20}$") #订单号(0开头的3～20位的整数)
    user_id: str= Field(min_length=1, max_length=64) # 用户id

class RefundSubmission(BaseModel): # 真正执行退款时的数据
    order_id: str= Field(pattern=r"^O\d{3,20}$")
    user_id: str= Field(min_length=1, max_length=64)
    amount: float= Field(gt=0, le=10_000) # 退款金额
    reason_code: Literal["DAMAGED_ITEM","OTHER"] # 退款原因
    idempotency_key: str= Field(min_length=16, max_length=128)
    approval: dict[str,Any] # 审批信息

    @field_validator("amount")
    @classmethod
    def two_decimal_places(cls, value: float) -> float: # 限制金额最多两位小数
        rounded = round(value, 2) # 把 value 四舍五入到小数点后 2 位
        if abs(value - rounded) > 1e-9:
            raise ValueError("金额最多保留两位小数")
        return rounded
    