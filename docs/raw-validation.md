# 真实源数据校验模块

## 定位

`fast_nc_zarr.raw_validation` 面向包含多个数据集子目录的真实数据根目录。它对每个子目录执行完整元数据检查、分层源值抽样，并可生成一个小范围 Zarr 进行转换冒烟验证，最终输出 JSON 报告。

入口：

```bash
pixi run validate-raw -- --help
python -m fast_nc_zarr.raw_validation --help
```

## 目录约定

```text
input-root/
├── dataset-a/
│   ├── file-001.nc
│   └── file-002.nc
└── dataset-b/
    ├── image.2001001.tif
    └── image.2001009.tif
```

模块只处理 `input-root` 的直接子目录；每个子目录被视为一个独立、同构的数据集。

## 工作流

对每个数据集：

1. 检查文件名字段和源 `time` 元数据，选择建议时间规则。
2. 若日期字段存在歧义，要求使用 `--time-field DATASET=INDEX` 明确指定。
3. 调用标准源检查服务，读取全部文件元数据。
4. 记录读取引擎、文件数、总字节数、时间范围、频率和缺失时间数。
5. 记录纬度、经度方向、范围、分辨率和规则性。
6. 记录变量维度、dtype 和原生 chunks。
7. 从时间和空间位置分层抽样源值。
8. 可选执行小范围文件名模式转换并验证输出。
9. 以临时文件写 JSON，再原子替换目标报告。

## 使用方式

只生成检查和抽样报告：

```bash
pixi run validate-raw -- \
  --input-root /data/RAW_DATA \
  --output /data/reports/raw-validation.json
```

同时进行转换冒烟验证：

```bash
pixi run validate-raw -- \
  --input-root /data/RAW_DATA \
  --output /data/reports/raw-validation.json \
  --workers 4 \
  --sample-files 9 \
  --smoke-output-root /fast-ssd/raw-smoke
```

指定歧义文件名中的完整日期字段：

```bash
pixi run validate-raw -- \
  --input-root /data/RAW_DATA \
  --output /data/reports/raw-validation.json \
  --time-field GLASS-PAR=3 \
  --time-field OTHER-DATASET=2
```

字段索引来自时间字段检查结果；同一个命令可以重复提供 `--time-field`。

## Python 入口

```python
from pathlib import Path
from fast_nc_zarr.raw_validation import validate_raw_tree

report = validate_raw_tree(
    Path("/data/RAW_DATA"),
    workers=4,
    sample_files=9,
    smoke_output_root=Path("/fast-ssd/raw-smoke"),
    time_field_overrides={"GLASS-PAR": 3},
)
```

## 报告内容

顶层报告包含 schema version、UTC 创建时间、输入根目录、总体状态、总文件数和总字节数。每个数据集条目包含：

- 模式和读取引擎；
- 文件数、总字节数和检查耗时；
- 时间数量、起止日期、频率和缺失数量；
- 经纬度轴统计；
- 变量定义；
- 分层源值样本；
- 可选转换冒烟结果。

任一数据集出现时间歧义、结构错误或转换错误时命令失败，不生成伪造的 passed 报告。

## 当前限制

- 只遍历输入根目录的直接子目录，不递归发现更深层的数据集。
- `sample_files` 必须为正整数；抽样不能替代完整逐值科学验证。
- 转换冒烟使用有界的小范围数据，不代表完整数据集的最终吞吐或空间需求。
- 模块继承转换器的日级时间、规则经纬网格和同构批次约束。
