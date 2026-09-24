# 项目结构与运行规范

## 目录职责

当前项目采用 `mllm-structure-style` 的配置驱动、原生 torchrun、策略插件化
结构。规范主干如下：

```text
configs/       Hydra 配置组：data、model、method、eval_tasks、env、paths
src/
  data/        只读取原始记录，不做 tokenizer 或模型处理
  models/      延迟加载模型、processor 和设备资源
  methods/     统一的 BaseMethod 与推理策略
  adapters/    VLMEvalKit、POPE runner 等外部边界
  utils/       分布式、日志、配置实例化和实验记录
  infer.py     唯一的纯推理编排入口
  eval.py      唯一的评测编排入口
tests/         contract、mock、配置和 adapter 测试
```

当前只保留 canonical 结构和已验证的 POPE runner。模型特定的完整推理逻辑仍集中在
当前 `scripts/pope_*` runner 中，由 `src/adapters/pope_runner.py` 统一调度；
不再保留旧的实验树、重复工具包或 vendored Transformers 构建副本。

## 常用命令

所有 Python 命令使用项目根目录 `.venv`，依赖由 `uv` 和 `uv.lock` 管理：

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/infer.py experiment=mock_smoke

uv run pytest -q
uv run ruff check src tests
```

真实 POPE 复现通过同一 Hydra 入口选择 `pope_runner` backend，模型和数据路径由配置或
环境变量注入，不写入 Python 代码。POPE runner 默认单进程运行；多进程
torchrun 只适用于 canonical `src/infer.py` 和支持分布式的评测 adapter：

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/eval.py experiment=pope_qwen25_vl method=vanilla method_name=vanilla

uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  src/eval.py experiment=pope_qwen25_vl method=only method_name=only
```

运行真实 CUDA 任务前仍须先使用 `gpu-scheduler` 检查设备和其他进程。模型、
POPE 文件和图像只从 `/data/lcq/.cache/huggingface/hub` 读取；预测和报告写入
Hydra 生成的 `logs/` 或显式输出目录。

VLMEvalKit、W&B 和 Optuna sweeper 都是可选依赖，分别使用
`uv sync --extra eval`、`uv sync --extra tracking` 和
`uv sync --extra sweeper` 安装，不改变默认推理环境。

## 扩展方法

新增方法只需：

1. 在 `src/methods/` 新建继承 `BaseMethod` 的策略类。
2. 在 `configs/method/` 新建对应配置并填写 `_target_`。
3. 添加至少一个 method contract 测试。

不要为算法分支修改 `src/infer.py`。模型特定的 processor、权重加载和设备放置
留在 `src/models/`；外部 benchmark 的 I/O 留在 `src/adapters/`。

## 复现记录

当前 POPE CLI runner 生成的全量结果和统一协议见根目录 `AGENTS.md`。
canonical mock smoke 只验证结构、配置组合、分布式收集和
方法契约，不替代真实模型质量实验。
