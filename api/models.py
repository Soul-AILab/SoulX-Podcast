"""
Pydantic Data Models for API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRequest(BaseModel):
    """生成请求模型（用于JSON body，配合文件上传使用）"""
    prompt_texts: List[str] = Field(
        ...,
        description="参考文本列表，长度应与上传的音频文件数量一致",
        min_items=1,
        max_items=4
    )
    dialogue_text: str = Field(
        ...,
        description="要生成的对话文本。单说话人直接输入文本，多说话人使用[S1][S2]标记",
        min_length=1
    )
    seed: Optional[int] = Field(
        default=1988,
        description="随机种子，用于复现结果"
    )
    temperature: Optional[float] = Field(
        default=0.6,
        ge=0.1,
        le=2.0,
        description="采样温度"
    )
    top_k: Optional[int] = Field(
        default=100,
        ge=1,
        le=500,
        description="Top-K采样参数"
    )
    top_p: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-P采样参数"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.25,
        ge=1.0,
        le=2.0,
        description="重复惩罚系数"
    )

    @validator('dialogue_text')
    def validate_dialogue_text(cls, v):
        """验证对话文本格式"""
        if not v.strip():
            raise ValueError("dialogue_text不能为空")
        return v.strip()


class TaskCreateResponse(BaseModel):
    """异步任务创建响应"""
    task_id: str = Field(..., description="任务唯一标识符")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    created_at: datetime = Field(..., description="任务创建时间")
    message: str = Field(default="任务已创建", description="提示信息")


class TaskStatusResponse(BaseModel):
    """任务状态查询响应"""
    task_id: str = Field(..., description="任务唯一标识符")
    status: TaskStatus = Field(..., description="任务状态")
    progress: Optional[int] = Field(None, ge=0, le=100, description="进度百分比")
    result_url: Optional[str] = Field(None, description="结果文件下载链接")
    error: Optional[str] = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="任务创建时间")
    started_at: Optional[datetime] = Field(None, description="任务开始时间")
    completed_at: Optional[datetime] = Field(None, description="任务完成时间")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "progress": 100,
                "result_url": "/download/123e4567-e89b-12d3-a456-426614174000.wav",
                "error": None,
                "created_at": "2025-11-01T12:00:00Z",
                "started_at": "2025-11-01T12:00:01Z",
                "completed_at": "2025-11-01T12:00:15Z"
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="healthy", description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    gpu_available: bool = Field(..., description="GPU是否可用")
    llm_engine: str = Field(..., description="当前使用的LLM引擎 (hf/vllm)")
    active_tasks: int = Field(default=0, description="正在处理的任务数")
    version: str = Field(default="1.0.0", description="API版本")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误详细信息")
    task_id: Optional[str] = Field(None, description="相关任务ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误发生时间")
