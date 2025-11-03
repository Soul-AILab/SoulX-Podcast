"""
API测试客户端示例
使用示例:
    python api/test_client.py --mode sync
    python api/test_client.py --mode async
"""
import requests
import time
import json
import argparse
from pathlib import Path


def test_sync_single_speaker(api_url: str):
    """测试同步生成 - 单说话人"""
    print("\n" + "=" * 60)
    print("测试: 同步生成 - 单说话人")
    print("=" * 60)

    # 准备文件
    audio_file = "example/audios/female_mandarin.wav"
    if not Path(audio_file).exists():
        print(f"错误: 找不到音频文件 {audio_file}")
        return

    files = {
        'prompt_audio': open(audio_file, 'rb')
    }
    data = {
        'prompt_texts': json.dumps(["喜欢攀岩、徒步、滑雪的语言爱好者。"]),
        'dialogue_text': '大家好，欢迎收听今天的节目。今天我们要聊一聊人工智能的最新进展。',
        'seed': 1988
    }

    print(f"发送请求到: {api_url}/generate")
    start_time = time.time()

    try:
        response = requests.post(f"{api_url}/generate", files=files, data=data)
        response.raise_for_status()

        # 保存结果
        output_path = "api/outputs/test_single_sync.wav"
        with open(output_path, 'wb') as f:
            f.write(response.content)

        elapsed = time.time() - start_time
        print(f"✓ 生成成功!")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  保存到: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        files['prompt_audio'].close()


def test_sync_multi_speaker(api_url: str):
    """测试同步生成 - 多说话人"""
    print("\n" + "=" * 60)
    print("测试: 同步生成 - 多说话人")
    print("=" * 60)

    # 准备文件
    audio_files = [
        "example/audios/female_mandarin.wav",
        "example/audios/male_mandarin.wav"
    ]

    for f in audio_files:
        if not Path(f).exists():
            print(f"错误: 找不到音频文件 {f}")
            return

    files = [
        ('prompt_audio', open(audio_files[0], 'rb')),
        ('prompt_audio', open(audio_files[1], 'rb'))
    ]
    data = {
        'prompt_texts': json.dumps([
            "喜欢攀岩、徒步、滑雪的语言爱好者。",
            "资深科技播客主持人。"
        ]),
        'dialogue_text': '[S1]大家好，欢迎收听今天的节目。[S2]是的，今天我们要聊聊人工智能。[S1]这个话题确实很有趣。',
        'seed': 1988
    }

    print(f"发送请求到: {api_url}/generate")
    start_time = time.time()

    try:
        response = requests.post(f"{api_url}/generate", files=files, data=data)
        response.raise_for_status()

        # 保存结果
        output_path = "api/outputs/test_multi_sync.wav"
        with open(output_path, 'wb') as f:
            f.write(response.content)

        elapsed = time.time() - start_time
        print(f"✓ 生成成功!")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  保存到: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        for _, file_obj in files:
            file_obj.close()


def test_async(api_url: str):
    """测试异步生成"""
    print("\n" + "=" * 60)
    print("测试: 异步生成")
    print("=" * 60)

    # 准备文件
    audio_files = [
        "example/audios/female_mandarin.wav",
        "example/audios/male_mandarin.wav"
    ]

    for f in audio_files:
        if not Path(f).exists():
            print(f"错误: 找不到音频文件 {f}")
            return

    files = [
        ('prompt_audio', open(audio_files[0], 'rb')),
        ('prompt_audio', open(audio_files[1], 'rb'))
    ]
    data = {
        'prompt_texts': json.dumps([
            "喜欢攀岩、徒步、滑雪的语言爱好者。",
            "资深科技播客主持人。"
        ]),
        'dialogue_text': '[S1]欢迎收听本期节目。[S2]今天的话题是AI语音合成。[S1]这确实是个很有意思的方向。[S2]没错，让我们深入探讨一下。',
        'seed': 1988
    }

    print(f"提交异步任务到: {api_url}/generate-async")

    try:
        # 提交任务
        response = requests.post(f"{api_url}/generate-async", files=files, data=data)
        response.raise_for_status()
        result = response.json()

        task_id = result['task_id']
        print(f"✓ 任务已创建: {task_id}")
        print(f"  状态: {result['status']}")
        print(f"  提示: {result['message']}")

        # 轮询任务状态
        print("\n等待任务完成...")
        max_attempts = 120  # 最多等待2分钟
        attempt = 0

        while attempt < max_attempts:
            time.sleep(2)
            attempt += 1

            status_response = requests.get(f"{api_url}/task/{task_id}")
            status_response.raise_for_status()
            status = status_response.json()

            print(f"  [{attempt}] 状态: {status['status']}, 进度: {status.get('progress', 0)}%")

            if status['status'] == 'completed':
                print(f"\n✓ 任务完成!")
                print(f"  耗时: {(attempt * 2):.0f}秒")

                # 下载结果
                download_url = f"{api_url}{status['result_url']}"
                print(f"  下载URL: {download_url}")

                audio_response = requests.get(download_url)
                audio_response.raise_for_status()

                output_path = "api/outputs/test_async.wav"
                with open(output_path, 'wb') as f:
                    f.write(audio_response.content)

                print(f"  保存到: {output_path}")
                break

            elif status['status'] == 'failed':
                print(f"\n✗ 任务失败: {status.get('error', '未知错误')}")
                break

        else:
            print(f"\n✗ 超时: 任务未在{max_attempts * 2}秒内完成")

    except requests.exceptions.RequestException as e:
        print(f"✗ 请求失败: {e}")
    finally:
        for _, file_obj in files:
            file_obj.close()


def test_health(api_url: str):
    """测试健康检查"""
    print("\n" + "=" * 60)
    print("测试: 健康检查")
    print("=" * 60)

    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        health = response.json()

        print(f"✓ API运行正常")
        print(f"  状态: {health['status']}")
        print(f"  模型已加载: {health['model_loaded']}")
        print(f"  GPU可用: {health['gpu_available']}")
        print(f"  活跃任务: {health['active_tasks']}")
        print(f"  版本: {health['version']}")

    except requests.exceptions.RequestException as e:
        print(f"✗ 健康检查失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="API测试客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API服务地址（默认: http://localhost:8000）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["health", "sync", "async", "all"],
        default="all",
        help="测试模式（默认: all）"
    )

    args = parser.parse_args()

    print("SoulX-Podcast API 测试客户端")
    print(f"API地址: {args.url}")

    # 确保输出目录存在
    Path("api/outputs").mkdir(parents=True, exist_ok=True)

    if args.mode in ["health", "all"]:
        test_health(args.url)

    if args.mode in ["sync", "all"]:
        test_sync_single_speaker(args.url)
        test_sync_multi_speaker(args.url)

    if args.mode in ["async", "all"]:
        test_async(args.url)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
