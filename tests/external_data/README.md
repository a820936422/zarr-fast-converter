# 外部数据受控样本测试

本目录统一保存外部数据测试代码、只读样本链接、运行日志、临时 Zarr 输出和 JSON 结果。

## 运行

```bash
# L0：受控样本正确性（每个数据集 1–2 个文件、32×32 窗口、≤2 时间步）
pixi run python tests/external_data/run_samples.py

# L1：v1.7.8 真实数据专项（T1–T5；T6 使用 pixi run validate-raw）
pixi run python tests/external_data/run_v178_l1.py
```

运行前复制 `manifest.example.json` 为 `manifest.local.json`，将 `source_root` 改为本机外部数据目录。当前机器的本地清单已配置好，未纳入 Git。

## 数据安全

- `inputs/` 只包含指向外部原始文件的符号链接，不复制、不写入、不删除源数据。
- 每类数据只选一到两个文件；转换只读取中心 `32×32` 空间窗口，最多两个时间步。
- 测试固定单 worker，结果用于发现后端兼容性和逻辑问题，不用于宣称绝对性能。

## 清理标记

以下目录和文件均为测试后可删除的本地生成物：

- `inputs/`：样本符号链接；
- `work/`：临时 Zarr 输出；
- `logs/`：每个数据集的执行日志；
- `results/`：JSON 测试结果；
- `manifest.local.json`：包含本机绝对路径的本地清单。

应保留并提交的内容：

- `run_samples.py`：受控样本测试代码；
- `run_v178_l1.py`：v1.7.8 L1 真实数据专项脚本（T1–T5）；
- `manifest.example.json`：不含本机绝对路径的清单模板；
- 本说明文件。
