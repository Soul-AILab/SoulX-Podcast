"""
FastAPI Main Application for SoulX-Podcast Voice Cloning API
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List
import json
import threading

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import scipy.io.wavfile as wavfile

from api.config import config
from api.models import (
    TaskCreateResponse,
    TaskStatusResponse,
    HealthResponse,
    ErrorResponse,
    TaskStatus,
)
from api.service import get_service
from api.tasks import get_task_manager
from api.utils import (
    generate_task_id,
    save_upload_file,
    validate_audio_files,
    validate_dialogue_format,
    cleanup_old_files,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建一个全局锁来控制同步推理的并发
inference_lock = threading.Lock()
active_inferences = 0
MAX_CONCURRENT_SYNC_INFERENCES = 1  # 限制同步推理的并发数


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting SoulX-Podcast API...")

    # 初始化模型（在主线程）
    logger.info("Loading model...")
    service = get_service()
    if not service.is_loaded():
        raise RuntimeError("Failed to load model")

    # 启动任务管理器
    task_manager = get_task_manager()
    task_manager.start_workers(config.max_concurrent_tasks)

    # 启动文件清理任务
    async def cleanup_task():
        while True:
            await asyncio.sleep(600)  # 每10分钟清理一次
            count = cleanup_old_files(config.temp_dir, config.file_cleanup_minutes)
            count += cleanup_old_files(config.output_dir, config.file_cleanup_minutes)
            if count > 0:
                logger.info(f"Cleaned up {count} old files")

    cleanup_task_handle = asyncio.create_task(cleanup_task())

    logger.info("API started successfully!")

    yield

    # 关闭时
    logger.info("Shutting down API...")
    cleanup_task_handle.cancel()

    # 快速关闭任务管理器
    try:
        await asyncio.wait_for(task_manager.shutdown(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Task manager shutdown timeout, forcing exit")

    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("API shutdown completed")


# 创建FastAPI应用
app = FastAPI(
    title="SoulX-Podcast Voice Cloning API",
    description="基于SoulX-Podcast的语音克隆API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """根路径"""
    return {
        "name": "SoulX-Podcast Voice Cloning API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    service = get_service()
    task_manager = get_task_manager()

    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded(),
        gpu_available=torch.cuda.is_available(),
        llm_engine=config.llm_engine,
        active_tasks=task_manager.get_active_task_count(),
        version="1.0.0"
    )


@app.post("/generate", tags=["Generation"])
async def generate_sync(
    prompt_audio: List[UploadFile] = File(..., description="参考音频文件（1-4个）"),
    prompt_texts: List[str] = Form(..., description="参考文本JSON数组，如: [\"文本1\", \"文本2\"]"),
    dialogue_text: str = Form(..., description="要生成的对话文本"),
    seed: int = Form(default=1988, description="随机种子"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
):
    """
    同步生成语音（直接返回音频文件）

    适用于短音频生成（预计<30秒）
    """
    task_id = generate_task_id()

    try:
        # 验证音频文件
        validate_audio_files(prompt_audio)

        # 解析prompt_texts
        try:
            prompt_text_list = prompt_texts
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts必须是JSON数组")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"prompt_texts JSON格式错误: {str(e)}")

        # 验证数量匹配
        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"参考音频数量({len(prompt_audio)})与参考文本数量({len(prompt_text_list)})不匹配"
            )

        # 验证对话格式
        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 保存上传的文件
        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        logger.info(f"Sync generation started: task_id={task_id}, speakers={len(audio_paths)}")

        # 调用服务生成
        service = get_service()
        sample_rate, audio_array = service.generate(
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # 保存结果
        output_filename = f"{task_id}.wav"
        output_path = config.output_dir / output_filename
        wavfile.write(str(output_path), sample_rate, audio_array)

        logger.info(f"Sync generation completed: task_id={task_id}")

        # 返回文件
        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=output_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-async", response_model=TaskCreateResponse, tags=["Generation"])
async def generate_async(
    prompt_audio: List[UploadFile] = File(..., description="参考音频文件（1-4个）"),
    prompt_texts: str = Form(..., description="参考文本JSON数组"),
    dialogue_text: str = Form(..., description="要生成的对话文本"),
    seed: int = Form(default=1988, description="随机种子"),
    temperature: float = Form(default=0.6, ge=0.1, le=2.0, description="采样温度"),
    top_k: int = Form(default=100, ge=1, le=500, description="Top-K采样"),
    top_p: float = Form(default=0.9, ge=0.0, le=1.0, description="Top-P采样"),
    repetition_penalty: float = Form(default=1.25, ge=1.0, le=2.0, description="重复惩罚"),
):
    """
    异步生成语音（返回任务ID）

    适用于长音频生成或批量任务
    """
    task_id = generate_task_id()

    try:
        # 验证音频文件
        validate_audio_files(prompt_audio)

        # 解析prompt_texts
        try:
            prompt_text_list = json.loads(prompt_texts)
            if not isinstance(prompt_text_list, list):
                raise ValueError("prompt_texts必须是JSON数组")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"prompt_texts JSON格式错误: {str(e)}")

        # 验证数量匹配
        if len(prompt_audio) != len(prompt_text_list):
            raise HTTPException(
                status_code=400,
                detail=f"参考音频数量({len(prompt_audio)})与参考文本数量({len(prompt_text_list)})不匹配"
            )

        # 验证对话格式
        is_valid, error_msg = validate_dialogue_format(dialogue_text, len(prompt_audio))
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 保存上传的文件
        audio_paths = []
        for i, file in enumerate(prompt_audio):
            path = save_upload_file(file, task_id, i)
            audio_paths.append(str(path))

        # 创建异步任务
        task_manager = get_task_manager()
        task = await task_manager.create_task(
            task_id=task_id,
            prompt_audio_paths=audio_paths,
            prompt_texts=prompt_text_list,
            dialogue_text=dialogue_text,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        logger.info(f"Async task created: task_id={task_id}")

        return TaskCreateResponse(
            task_id=task_id,
            status=task.status,
            created_at=task.created_at,
            message=f"任务已创建，当前队列中有 {task_manager.queue.qsize()} 个任务"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """查询任务状态"""
    task_manager = get_task_manager()
    task = task_manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 构建结果URL
    result_url = None
    if task.status == TaskStatus.COMPLETED and task.result_path:
        result_url = f"/download/{task.result_path.name}"

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        result_url=result_url,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


@app.get("/download/{filename}", tags=["Download"])
async def download_file(filename: str):
    """下载生成的音频文件"""
    file_path = config.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        log_level="info"
    )
