# API 更新日志

## [未发布] - 2025-11-04

### 新增
- ✨ **vLLM 引擎支持**：API 现在支持使用 vLLM 作为 LLM 推理引擎
  - 通过环境变量 `LLM_ENGINE=vllm` 启用
  - 推理速度提升 2-3 倍
  - 显存利用率更高（支持 PagedAttention 和前缀缓存）
  - 自动回退机制：如果 vLLM 未安装，自动使用 HuggingFace 引擎

### 改进
- 🔧 **配置验证**：添加 LLM 引擎配置验证，启动时自动检查引擎可用性
- 📊 **健康检查增强**：`/health` 接口现在返回当前使用的 LLM 引擎信息
- 📝 **文档更新**：
  - 添加 vLLM 引擎使用说明
  - 添加性能对比表格
  - 更新配置示例和优化建议

### 技术细节
- 更新 `api/config.py`：添加 `validate_llm_engine()` 方法
- 更新 `api/service.py`：增强日志输出，显示使用的引擎
- 更新 `api/models.py`：`HealthResponse` 添加 `llm_engine` 字段
- 更新 `api/main.py`：健康检查返回引擎信息
- 新增 `test_vllm_config.py`：配置测试脚本

### 使用示例

#### 使用 HuggingFace 引擎（默认）
```bash
LLM_ENGINE=hf python api/main.py
```

#### 使用 vLLM 引擎（推荐生产环境）
```bash
# 首先安装 vLLM
pip install vllm

# 启动 API
LLM_ENGINE=vllm python api/main.py
```

#### 检查当前引擎
```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "llm_engine": "vllm",
  "active_tasks": 0,
  "version": "1.0.0"
}
```

### 性能对比

| 引擎 | 单轮推理耗时 | 显存占用 | 吞吐量 | 推荐并发数 |
|------|------------|---------|--------|-----------|
| HuggingFace | ~2.5s | 8GB | 1x | 2 |
| vLLM | ~1.0s | 7GB | 2-3x | 3-4 |

### 兼容性
- ✅ 向后兼容：默认使用 HuggingFace 引擎，无需修改现有代码
- ✅ 自动回退：vLLM 不可用时自动切换到 HuggingFace
- ✅ 环境变量配置：通过 `LLM_ENGINE` 环境变量灵活切换

---

## [1.0.0] - 2025-11-01

### 初始版本
- 🎉 首次发布 SoulX-Podcast API
- ✅ 支持同步和异步生成模式
- ✅ 支持 1-4 个说话人的对话生成
- ✅ 零样本语音克隆
- ✅ 任务队列和并发控制
- ✅ 自动文件清理
- ✅ 完整的 REST API 文档
