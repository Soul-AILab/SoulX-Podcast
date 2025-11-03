"""
Quick start script for SoulX-Podcast API
使用示例:
    python run_api.py
    python run_api.py --port 8080
    python run_api.py --model pretrained_models/SoulX-Podcast-1.7B-dialect
"""
import os
import sys
import argparse
import signal
import time


def main():
    parser = argparse.ArgumentParser(description="启动SoulX-Podcast API服务")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained_models/SoulX-Podcast-1.7B",
        help="模型路径（默认: pretrained_models/SoulX-Podcast-1.7B）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API端口（默认: 8000）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API主机地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="LLM引擎（默认: hf）"
    )
    parser.add_argument(
        "--fp16-flow",
        action="store_true",
        help="使用FP16精度的Flow模型（更快但略降质量）"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=2,
        help="最大并发任务数（默认: 2）"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载（开发模式）"
    )

    args = parser.parse_args()

    # 设置环境变量
    os.environ["MODEL_PATH"] = args.model
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["LLM_ENGINE"] = args.engine
    os.environ["FP16_FLOW"] = "true" if args.fp16_flow else "false"
    os.environ["MAX_CONCURRENT_TASKS"] = str(args.max_tasks)
    os.environ["API_RELOAD"] = "true" if args.reload else "false"

    # 检查模型路径
    if not os.path.exists(args.model):
        print(f"错误: 模型路径不存在: {args.model}")
        print("\n请先下载模型:")
        print(f"huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B --local-dir {args.model}")
        sys.exit(1)

    # 打印启动信息
    print("=" * 60)
    print("SoulX-Podcast API 服务启动中...")
    print("=" * 60)
    print(f"模型路径: {args.model}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://localhost:{args.port}/docs")
    print(f"LLM引擎: {args.engine}")
    print(f"FP16 Flow: {'是' if args.fp16_flow else '否'}")
    print(f"最大并发: {args.max_tasks}")
    print("=" * 60)
    print("\n正在加载模型，请稍候...\n")
    print("提示: 按 Ctrl+C 可以停止服务（如果响应慢，连按两次强制退出）\n")

    # 设置信号处理器，支持快速退出
    shutdown_count = 0

    def signal_handler(signum, frame):
        nonlocal shutdown_count
        shutdown_count += 1
        if shutdown_count == 1:
            print("\n\n正在优雅关闭服务... (再按一次 Ctrl+C 强制退出)")
        else:
            print("\n\n强制退出!")
            # 清理GPU内存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # 启动API
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
