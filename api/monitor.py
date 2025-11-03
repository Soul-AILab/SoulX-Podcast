"""
GPU Memory Monitor for SoulX-Podcast API
用于监控GPU内存使用情况，帮助诊断内存泄漏问题
"""
import time
import torch
import psutil
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return None

    info = {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'free_gb': (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
    }
    return info


def get_system_memory_info():
    """获取系统内存信息"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / 1024**3,
        'available_gb': memory.available / 1024**3,
        'used_gb': memory.used / 1024**3,
        'percent': memory.percent
    }


def monitor_memory(interval=5):
    """持续监控内存使用"""
    logger.info("Starting memory monitor...")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            # GPU内存
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                logger.info(
                    f"GPU Memory - Allocated: {gpu_info['allocated_gb']:.2f}GB, "
                    f"Reserved: {gpu_info['reserved_gb']:.2f}GB, "
                    f"Free: {gpu_info['free_gb']:.2f}GB, "
                    f"Max: {gpu_info['max_allocated_gb']:.2f}GB"
                )

            # 系统内存
            sys_info = get_system_memory_info()
            logger.info(
                f"System Memory - Used: {sys_info['used_gb']:.2f}GB / "
                f"{sys_info['total_gb']:.2f}GB ({sys_info['percent']:.1f}%)"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Memory monitor stopped")


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("GPU cache cleared")

        # 显示清理后的状态
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            logger.info(
                f"After clearing - Allocated: {gpu_info['allocated_gb']:.2f}GB, "
                f"Reserved: {gpu_info['reserved_gb']:.2f}GB"
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_gpu_cache()
    else:
        monitor_memory()