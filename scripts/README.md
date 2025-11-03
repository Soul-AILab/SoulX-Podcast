# Scripts 目录

本目录包含各种测试和工具脚本。

## 测试脚本

### test_vllm_config.py

测试 vLLM 配置和引擎切换功能。

**用途**：
- 验证 API 配置是否正确
- 检查 vLLM 是否已安装
- 测试引擎切换功能
- 显示当前配置摘要

**运行方法**：
```bash
python scripts/test_vllm_config.py
```

**输出示例**：
```
============================================================
测试 API 配置
============================================================

[OK] HF 引擎配置: hf
[OK] vLLM 引擎配置: vllm

============================================================
检查 vLLM 安装状态
============================================================

[OK] vLLM 已安装 (版本: 0.3.0)

============================================================
检查 LLM 引擎模块
============================================================

[OK] HFLLMEngine 可用
[OK] SUPPORT_VLLM = True
[OK] VLLMEngine 可用

============================================================
配置总结
============================================================

当前配置：
- 模型路径: pretrained_models/SoulX-Podcast-1.7B
- LLM 引擎: vllm
- FP16 Flow: False
- 最大并发任务: 2

引擎支持：
- HuggingFace: [OK] 可用
- vLLM: [OK] 可用

推荐配置：

# 使用 vLLM 引擎（推荐生产环境）
LLM_ENGINE=vllm \
MAX_CONCURRENT_TASKS=3 \
python api/main.py
```

### test_singleton.py

测试服务单例模式实现。

**用途**：
- 验证 SoulXPodcastService 单例模式
- 确保模型只加载一次
- 测试线程安全性

**运行方法**：
```bash
python scripts/test_singleton.py
```

## 工具脚本

### download.py

下载模型文件的辅助脚本。

**运行方法**：
```bash
python scripts/download.py
```

### run_api.py

API 服务启动脚本（带配置选项）。

**运行方法**：
```bash
python scripts/run_api.py
```

### setup_wsl.sh

WSL 环境设置脚本。

**运行方法**：
```bash
bash scripts/setup_wsl.sh
```

## 安装文件

### Miniforge3-Linux-x86_64.sh

Miniforge3 安装程序（Linux x86_64）。

### mambaforge.sh

Mambaforge 安装脚本。

## 其他脚本

### 计划添加的脚本

- `benchmark_engines.py` - 对比 HF 和 vLLM 引擎性能
- `test_api_endpoints.py` - API 端点集成测试
- `stress_test.py` - 压力测试脚本
- `cleanup_files.py` - 手动清理临时文件

## 使用建议

1. **开发阶段**：运行测试脚本验证配置
2. **部署前**：运行所有测试确保功能正常
3. **性能调优**：使用 benchmark 脚本对比不同配置

## 注意事项

- 所有脚本应从项目根目录运行
- 某些脚本可能需要模型文件已下载
- Windows 用户注意编码问题（脚本已处理）
