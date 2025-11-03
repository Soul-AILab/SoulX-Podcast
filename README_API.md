# SoulX-Podcast Voice Cloning API

基于SoulX-Podcast模型的语音克隆REST API服务。

## 功能特性

- ✅ **统一接口设计** - 单/多说话人共用同一套API
- ✅ **同步+异步双模式** - 支持实时返回和异步任务队列
- ✅ **零样本语音克隆** - 3-10秒参考音频即可克隆音色
- ✅ **多说话人对话** - 支持2-4个说话人的播客式对话生成
- ✅ **文件上传/下载** - 标准的multipart/form-data接口
- ✅ **并发控制** - 防止GPU过载的智能队列管理
- ✅ **自动清理** - 临时文件自动过期删除

## 快速开始

### 1. 安装依赖

```bash
# 安装项目基础依赖（如果还未安装）
cd SoulX-Podcast
pip install -r requirements.txt

# 安装API额外依赖
pip install -r api/requirements.txt
```

### 2. 下载模型

```bash
# 下载基础模型（不含方言）
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B --local-dir pretrained_models/SoulX-Podcast-1.7B
```

### 3. 启动API服务

```bash
# 基础启动
python api/main.py

# 或使用环境变量配置
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B \
LLM_ENGINE=hf \
MAX_CONCURRENT_TASKS=2 \
python api/main.py
```

服务启动后访问：
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## API接口说明

### 1. 同步生成 `POST /generate`

适用于短音频（预计<30秒），直接返回音频文件。

**请求参数**：
```
multipart/form-data:
- prompt_audio: 音频文件列表（1-4个）
- prompt_texts: JSON数组，如 ["文本1", "文本2"]
- dialogue_text: 要生成的对话文本
- seed: 随机种子（默认1988）
- temperature: 采样温度（默认0.6）
- top_k: Top-K采样（默认100）
- top_p: Top-P采样（默认0.9）
- repetition_penalty: 重复惩罚（默认1.25）
```

**响应**：直接返回 `audio/wav` 文件

**示例 - 单说话人**：
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt_audio=@example/audios/female_mandarin.wav" \
  -F 'prompt_texts=["喜欢攀岩、徒步、滑雪的语言爱好者。"]' \
  -F 'dialogue_text=大家好，欢迎收听今天的节目。' \
  --output output.wav
```

**示例 - 多说话人**：
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt_audio=@example/audios/female_mandarin.wav" \
  -F "prompt_audio=@example/audios/male_mandarin.wav" \
  -F 'prompt_texts=["女主持人参考文本", "男主持人参考文本"]' \
  -F 'dialogue_text=[S1]大家好，欢迎收听今天的节目。[S2]是的，今天我们要聊一个有趣的话题。' \
  --output dialogue.wav
```

### 2. 异步生成 `POST /generate-async`

适用于长音频或批量任务，返回任务ID。

**请求参数**：同 `/generate`

**响应**：
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "created_at": "2025-11-01T12:00:00Z",
  "message": "任务已创建，当前队列中有 2 个任务"
}
```

**示例**：
```bash
curl -X POST "http://localhost:8000/generate-async" \
  -F "prompt_audio=@example/audios/female_mandarin.wav" \
  -F "prompt_audio=@example/audios/male_mandarin.wav" \
  -F 'prompt_texts=["女主持人参考文本", "男主持人参考文本"]' \
  -F 'dialogue_text=[S1]欢迎收听...[S2]今天我们...'
```

### 3. 查询任务状态 `GET /task/{task_id}`

**响应**：
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "progress": 100,
  "result_url": "/download/123e4567-e89b-12d3-a456-426614174000.wav",
  "error": null,
  "created_at": "2025-11-01T12:00:00Z",
  "started_at": "2025-11-01T12:00:01Z",
  "completed_at": "2025-11-01T12:00:15Z"
}
```

**任务状态说明**：
- `pending`: 等待处理
- `processing`: 正在生成
- `completed`: 已完成
- `failed`: 失败

**示例**：
```bash
curl "http://localhost:8000/task/123e4567-e89b-12d3-a456-426614174000"
```

### 4. 下载文件 `GET /download/{filename}`

**示例**：
```bash
curl "http://localhost:8000/download/123e4567-e89b-12d3-a456-426614174000.wav" \
  --output result.wav
```

### 5. 健康检查 `GET /health`

**响应**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "llm_engine": "vllm",
  "active_tasks": 2,
  "version": "1.0.0"
}
```

## 配置说明

通过环境变量配置：

```bash
# 模型路径
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B

# LLM引擎（hf 或 vllm）
LLM_ENGINE=hf

# 使用FP16精度的Flow模型（更快但略降质量）
FP16_FLOW=false

# API服务配置
API_HOST=0.0.0.0
API_PORT=8000

# 最大并发任务数（防止GPU过载）
MAX_CONCURRENT_TASKS=2
```

### LLM引擎选择

API 支持两种 LLM 推理引擎：

#### 1. HuggingFace (hf) - 默认引擎

```bash
LLM_ENGINE=hf python api/main.py
```

**特点**：
- ✅ 无需额外安装依赖
- ✅ 稳定性好，兼容性强
- ✅ 支持自定义 RAS (Repetition-Aware Sampling)
- ⚠️ 推理速度较慢

**适用场景**：开发测试、小规模部署

#### 2. vLLM - 高性能引擎

```bash
# 首先安装vLLM
pip install vllm

# 启动API
LLM_ENGINE=vllm python api/main.py
```

**特点**：
- ✅ 推理速度快（2-3倍提升）
- ✅ 显存利用率高（支持 PagedAttention）
- ✅ 支持前缀缓存（prefix caching）
- ⚠️ 需要额外安装 vLLM 库
- ⚠️ 对 GPU 驱动版本有要求

**适用场景**：生产环境、高并发场景

**性能对比**：

| 引擎 | 单轮推理耗时 | 显存占用 | 吞吐量 |
|------|------------|---------|--------|
| HuggingFace | ~2.5s | 8GB | 1x |
| vLLM | ~1.0s | 7GB | 2-3x |

**注意事项**：
- vLLM 需要 CUDA 11.8+ 和对应的 PyTorch 版本
- 如果 vLLM 未安装或加载失败，API 会自动回退到 HuggingFace 引擎
- 可以通过 `/health` 接口查看当前使用的引擎

## 对话文本格式

### 单说话人

直接输入文本即可：
```
大家好，欢迎收听今天的节目。
```

### 多说话人（2-4人）

使用 `[S1]`, `[S2]`, `[S3]`, `[S4]` 标记说话人：
```
[S1]大家好，欢迎收听今天的节目。
[S2]是的，今天我们要聊一个有趣的话题。
[S1]那我们开始吧！
```

**重要**：说话人编号必须与上传的音频顺序对应！
- 第1个音频 → [S1]
- 第2个音频 → [S2]
- 依此类推

## 副语言特征

在文本中插入特殊标记来控制情感：

```
[S1]最近活得特别赛博朋克哈！<|laughter|>现在连我妈都用AI写广场舞文案了。
[S2]这个例子很生动啊。<|sigh|>是的，特别是生成式AI。
```

支持的标记：
- `<|laughter|>` - 笑声
- `<|sigh|>` - 叹息

## Python客户端示例

```python
import requests

# 同步生成
files = {
    'prompt_audio': [
        open('speaker1.wav', 'rb'),
        open('speaker2.wav', 'rb')
    ]
}
data = {
    'prompt_texts': '["参考文本1", "参考文本2"]',
    'dialogue_text': '[S1]你好[S2]你好',
    'seed': 1988
}

response = requests.post('http://localhost:8000/generate', files=files, data=data)
with open('output.wav', 'wb') as f:
    f.write(response.content)

# 异步生成
response = requests.post('http://localhost:8000/generate-async', files=files, data=data)
task_id = response.json()['task_id']

# 查询状态
status = requests.get(f'http://localhost:8000/task/{task_id}').json()
print(status)

# 下载结果
if status['status'] == 'completed':
    audio = requests.get(f'http://localhost:8000{status["result_url"]}')
    with open('result.wav', 'wb') as f:
        f.write(audio.content)
```

## 性能优化建议

### 单说话人 vs 多说话人算力对比

根据分析，多说话人的计算成本约为单说话人的 **3.5-4.5倍**：

| 场景 | 预计耗时 | 主要瓶颈 |
|------|---------|---------|
| 单说话人（15s音频） | ~14s | LLM生成 |
| 双说话人（5轮对话） | ~50-65s | LLM生成（上下文×2） |

**优化建议**：
1. **使用 vLLM 引擎**：推理速度提升 2-3 倍
2. 短音频（<15秒）使用同步接口
3. 长对话（>30秒）使用异步接口
4. 合理设置 `MAX_CONCURRENT_TASKS`（HF引擎推荐2，vLLM引擎可设置3-4）
5. 启用 `FP16_FLOW=true` 可进一步提升速度（略微降低质量）

### 批量处理

```python
# 批量提交异步任务
task_ids = []
for dialogue in dialogues:
    resp = requests.post('http://localhost:8000/generate-async', ...)
    task_ids.append(resp.json()['task_id'])

# 批量查询状态
for task_id in task_ids:
    status = requests.get(f'http://localhost:8000/task/{task_id}').json()
    if status['status'] == 'completed':
        # 下载结果
        ...
```

## 故障排查

### 模型加载失败

```
RuntimeError: 模型加载失败
```

**解决方案**：
1. 检查 `MODEL_PATH` 是否正确
2. 确认模型文件已完整下载
3. 检查GPU显存是否充足（需要约8GB）

### 任务一直pending

**原因**：队列已满或GPU被占用

**解决方案**：
1. 查看 `/health` 接口的 `active_tasks`
2. 等待当前任务完成
3. 增加 `MAX_CONCURRENT_TASKS`（如果显存足够）

### 文件格式错误

```
HTTPException: 文件格式不支持
```

**解决方案**：
支持的音频格式：`.wav`, `.mp3`, `.flac`, `.m4a`

建议使用 16kHz/24kHz 的 WAV 格式

## 技术架构

```
FastAPI (Web框架)
    ├── Service Layer (模型封装)
    │   └── SoulXPodcast Model
    │       ├── S3Tokenizer (音频分词)
    │       ├── Qwen3 LLM (文本→语音token)
    │       ├── Flow Model (token→mel)
    │       └── HiFi-GAN (mel→wav)
    │
    ├── Task Manager (异步队列)
    │   ├── asyncio.Queue
    │   ├── Semaphore (并发控制)
    │   └── Worker Threads
    │
    └── Utils (工具函数)
        ├── 文件验证
        ├── 格式解析
        └── 自动清理
```

## 许可证

Apache 2.0 - 详见项目根目录的 [LICENSE](../LICENSE) 文件

## 联系与支持

- 技术问题：提交 GitHub Issue
- 更多示例：访问 https://soul-ailab.github.io/soulx-podcast/
- 项目主页：https://github.com/Soul-AILab/SoulX-Podcast
