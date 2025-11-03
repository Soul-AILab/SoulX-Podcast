"""
Utility Functions for API
"""
import os
import re
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
from fastapi import UploadFile, HTTPException
import logging

from api.config import config

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """生成唯一的任务ID"""
    return str(uuid.uuid4())


def save_upload_file(upload_file: UploadFile, task_id: str, index: int) -> Path:
    """
    保存上传的文件到临时目录

    Args:
        upload_file: FastAPI上传文件对象
        task_id: 任务ID
        index: 文件索引

    Returns:
        Path: 保存的文件路径
    """
    try:
        # 获取文件扩展名
        file_extension = Path(upload_file.filename).suffix or ".wav"

        # 构建保存路径
        filename = f"{task_id}_prompt_{index}{file_extension}"
        file_path = config.temp_dir / filename

        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        logger.info(f"Saved upload file to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save upload file: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")
    finally:
        upload_file.file.close()


def validate_audio_files(files: List[UploadFile]) -> None:
    """
    验证音频文件

    Args:
        files: 上传的文件列表

    Raises:
        HTTPException: 如果验证失败
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="至少需要上传1个参考音频文件")

    if len(files) > 4:
        raise HTTPException(status_code=400, detail="最多支持4个说话人（4个音频文件）")

    # 验证文件类型和大小
    allowed_extensions = {".wav", ".mp3", ".flac", ".m4a"}
    for i, file in enumerate(files):
        # 检查文件扩展名
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 格式不支持。支持的格式: {', '.join(allowed_extensions)}"
            )

        # 检查文件大小（通过content-length header，可能不准确）
        if hasattr(file, 'size') and file.size and file.size > config.max_upload_size:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 超过最大大小限制 ({config.max_upload_size / 1024 / 1024}MB)"
            )


def validate_dialogue_format(dialogue_text: str, num_speakers: int) -> Tuple[bool, str]:
    """
    验证对话文本格式

    Args:
        dialogue_text: 对话文本
        num_speakers: 说话人数量

    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    dialogue_text = dialogue_text.strip()

    # 单说话人模式：不需要特殊格式
    if num_speakers == 1:
        if len(dialogue_text) == 0:
            return False, "对话文本不能为空"
        # 单说话人可以不使用[S1]标记
        return True, ""

    # 多说话人模式：需要[S1][S2]等标记
    # 提取所有说话人标记
    speaker_pattern = r'\[S[1-4]\]'
    matches = re.findall(speaker_pattern, dialogue_text)

    if not matches:
        return False, f"多说话人模式需要使用说话人标记，如: [S1]你好[S2]你好"

    # 检查使用的说话人ID是否在有效范围内
    used_speakers = set()
    for match in matches:
        speaker_id = int(match[2])  # 提取[S1]中的1
        used_speakers.add(speaker_id)

        if speaker_id > num_speakers:
            return False, f"文本中使用了说话人[S{speaker_id}]，但只提供了{num_speakers}个参考音频"

    return True, ""


def cleanup_old_files(directory: Path, minutes: int = 30) -> int:
    """
    清理过期的文件

    Args:
        directory: 要清理的目录
        minutes: 文件保留时间（分钟）

    Returns:
        int: 清理的文件数量
    """
    if not directory.exists():
        return 0

    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    cleaned_count = 0

    try:
        for file_path in directory.glob("*"):
            if file_path.is_file():
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count


def format_audio_duration(seconds: float) -> str:
    """格式化音频时长"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_dialogue_text(dialogue_text: str, num_speakers: int) -> List[str]:
    """
    解析对话文本为列表格式

    Args:
        dialogue_text: 原始对话文本
        num_speakers: 说话人数量

    Returns:
        List[str]: 分段的对话列表，每段包含说话人标记
    """
    # 单说话人：直接添加[S1]标记
    if num_speakers == 1:
        if not dialogue_text.startswith("[S1]"):
            return [f"[S1]{dialogue_text}"]
        else:
            return [dialogue_text]

    # 多说话人：按[S1][S2]分割
    pattern = r'(\[S[1-4]\][^\[\]]*)'
    segments = re.findall(pattern, dialogue_text)

    # 过滤空段落
    segments = [seg.strip() for seg in segments if seg.strip()]

    return segments
