# -*- coding: utf-8 -*-
"""
File management utilities for SoulX-Podcast WebUI.
Handles config file operations and ZIP file creation.
"""

import os
import json
import zipfile
from datetime import datetime
from typing import List, Optional

from .constants import CONFIG_DIR
from .i18n import i18n


# =============================================================================
# Config Directory Operations
# =============================================================================

def ensure_config_dir():
    """Ensure the config directory exists."""
    os.makedirs(CONFIG_DIR, exist_ok=True)


def list_config_files() -> List[str]:
    """返回 config/ 下的 JSON 文件名列表（仅文件名，不含路径）。"""
    ensure_config_dir()
    try:
        files = [f for f in os.listdir(CONFIG_DIR) if f.lower().endswith(".json")]
    except Exception:
        files = []
    # 按文件名倒序（通常包含时间戳）
    return sorted(files, reverse=True)


def read_json_file(path: str) -> dict:
    """Read a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, data: dict):
    """Write data to a JSON file atomically."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# =============================================================================
# ZIP File Operations
# =============================================================================

def create_zip_file(
    file_list: List[str], 
    output_dir: str, 
    timestamp: str = None, 
    file_number: int = None
) -> Optional[str]:
    """
    创建包含所有文件的 zip 压缩包
    
    Args:
        file_list: 要打包的文件路径列表
        output_dir: 输出目录
        timestamp: 时间戳（如果不提供则自动生成）
        file_number: 文件序号（用于文件名前缀）
    
    Returns:
        zip 文件路径，如果失败则返回 None
    """
    if not file_list:
        return None
    
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 如果提供了文件序号，则在文件名前添加序号前缀
        if file_number is not None:
            zip_filename = os.path.join(output_dir, f"{file_number:03d}_all_audio_files.zip")
        else:
            zip_filename = os.path.join(output_dir, "all_audio_files.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_list:
                if os.path.exists(file_path):
                    # 只保存文件名，不包含完整路径
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    print(f"[INFO] {i18n('log_file_added_to_zip').format(filename=arcname)}")
        
        print(f"[INFO] {i18n('log_zip_created').format(filename=zip_filename)}")
        return zip_filename
    except Exception as e:
        print(f"[ERROR] {i18n('log_error_creating_zip').format(error=str(e))}")
        import traceback
        traceback.print_exc()
        return None


def create_all_zip(base_output_dir: str, all_files: List[str]) -> Optional[str]:
    """
    创建包含所有任务文件的 all.zip 压缩包，保持目录结构
    
    Args:
        base_output_dir: 基础输出目录（时间戳文件夹）
        all_files: 所有要打包的文件路径列表
    
    Returns:
        all.zip 文件路径，如果失败则返回 None
    """
    if not all_files:
        return None
    
    try:
        zip_filename = os.path.join(base_output_dir, "all.zip")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in all_files:
                if os.path.exists(file_path):
                    # 保持相对路径结构，相对于base_output_dir
                    # 使用 os.path.relpath 更安全，可以处理各种路径格式
                    try:
                        arcname = os.path.relpath(file_path, base_output_dir)
                        # 确保路径使用正斜杠（zip文件标准）
                        arcname = arcname.replace(os.sep, '/')
                    except ValueError:
                        # 如果文件不在同一驱动器上（Windows），使用文件名
                        arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    print(f"[INFO] 已添加文件到 all.zip: {arcname}")
        
        print(f"[INFO] 已创建 all.zip: {zip_filename}")
        return zip_filename
    except Exception as e:
        print(f"[ERROR] 创建 all.zip 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

