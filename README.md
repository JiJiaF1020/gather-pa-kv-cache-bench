# Ascend 950 GatherPaKvCache 带宽测试

本仓对 CANN 内置算子 `GatherPaKvCache`（torch_npu 封装 `torch_npu.npu_gather_pa_kv_cache`）做正确性校验与 DRAM/HBM→HBM 带宽扫参。算子来自 CANN ops-transformer（<https://gitcode.com/cann/ops-transformer/tree/master/attention/gather_pa_kv_cache>），由 torch_npu 自带，**本仓不需要编译任何自定义算子**（无 csrc/torch_extension）。测试方法论、扫参纪律和六列带宽口径完全复刻 `nanovllm-scatter_copy`（scatter_copy_a5_c8 分支）。

## 算子与接口

调用方式与 vLLM-ascend 的 `kv_cache_load`（`vllm_ascend/device/device_op.py`）一致：

```python
import torch
import torch_npu

torch_npu.npu_gather_pa_kv_cache(
    key_cache,      # [num_blocks, 128, num_heads, head_size_k]，Norm 模式 ND，只读
    value_cache,    # [num_blocks, 128, num_heads, head_size_v]，只读
    block_tables,   # int32[batch, source_blocks]，请求逻辑块 -> 源物理块
    seq_lens,       # int32[batch]，每个请求的固定 token 数（per-request 长度，非 cumsum）
    seq_offset=seq_offset,  # int32[batch]，本测试固定全零
    key=key_out,    # caller 分配的输出 [total_tokens, num_heads, head_size_k]
    value=value_out,        # [total_tokens, num_heads, head_size_v]
)
```

所有 tensor 位于同一 NPU、连续存储。输出为紧凑布局，`total_tokens = sum(seq_lens)`；本测试中每个请求的 `seq_lens` 相同（等于 `copy_count`），故 `total_tokens = batch_size × copy_count`。

默认布局对齐 C8 packed MLA 行：`block_size=128`、`num_heads=1`、`head_size_k=512`（fp8_e4m3fn，512B latent）、`head_size_v=72`（bf16，128B RoPE + 16B scale 打包成 8 个 bf16），合计 **656 字节/token**，与 C8 scatter_copy 的 `ROW_BYTES=656` 口径完全一致。布局可用 `--num-heads --head-size-k --head-size-v` 覆盖；bytes/token 始终由配置计算，不硬编码。fp8/bf16 源张量经 int8/int16 存储初始化后 `.view()` 得到，正确性比较按位进行（避免 NaN != NaN 问题）。

## 环境要求

- Ascend 950（默认校验设备名，调试可用 `--allow-non-a5` 跳过）。
- CANN 工具包（`ASCEND_HOME_PATH`，默认 `/usr/local/Ascend/ascend-toolkit/latest`）与匹配的 PyTorch/torch_npu；torch_npu 需支持 `float8_e4m3fn` 与 `empty_with_swapped_memory`。
- 无自定义算子编译步骤；无需设置 `PYTHONPATH` 指向任何 torch_extension。

## 统一带宽测试

在仓库根目录、CANN/PyTorch 环境中运行：

```bash
A5_TEST_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CARD_COUNTS="1 2 4 8" BATCH_SIZES="8 16 24 32" bash tests/run_gather_pa_kv_cache_multiprocess.sh
```

只跑一组：

```bash
bash tests/run_gather_pa_kv_cache_multiprocess.sh \
  --cards 8 --batch-size 8 --copy-count 300
```

参数也支持多个值，例如 `--cards 1 2 4 --batch-size 8 24 --copy-count 200 300`。命令行参数优先于环境变量。`--dry-run`（或 `DRY_RUN=1`）只检查参数并打印每组命令，不加载 CANN/Torch、不启动 NPU，也不写结果文件，可在无 CANN 的开发机上使用：

```bash
bash tests/run_gather_pa_kv_cache_multiprocess.sh \
  --visible-devices 2,3,6,7 --cards 1 2 4 \
  --batch-size 8 16 --copy-count 0 200 300 --dry-run
```

参数约定：

- `cards` / `CARD_COUNTS` 统计当前可见列表中的**逻辑 NPU/die 数**。每组使用前 N 个可见设备；每个 die 启动独立进程，用文件屏障同步 warmup 与计时起点。
- 设备可见性优先级：`TEST_VISIBLE_DEVICES` > `A5_TEST_VISIBLE_DEVICES` > 已有 `ASCEND_RT_VISIBLE_DEVICES` > 默认 `0,1,2,3,4,5,6,7`；`--visible-devices` 可覆盖。
- `batch_size` 是**每 die** 的请求数；`copy_count` 是**每个请求**固定 gather 的 token 数（即 `seq_lens` 的固定值）。每 die 每轮有效 payload = `batch_size × copy_count × bytes_per_token`。
- 默认卡数为可见数量内的 `1 2 4 8 16`；显式指定时支持任意正数卡数，不局限于这些档位。
- 默认 `BATCH_SIZES="8 16 24 32"`、`COPY_COUNTS="0 100 200 300 500 2048"`、`SOURCE_LEN=65536`、`WARMUP=10`、`ITERS=1000`、`SEED=7`。
- 其余参数：`--source-len`、`--num-heads`、`--head-size-k`、`--head-size-v`、`--block-size`、`--source-memory {dram,hbm}`、`--warmup`、`--iters`、`--seed`、`--results-dir`；同名大写下划线环境变量（`SOURCE_LEN`/`BLOCK_SIZE`/`WARMUP`/`ITERS`/`SEED`/`SOURCE_MEMORY`/`RESULTS_DIR`）也可使用。Python 路径可用 `PYTHON_BIN` 指定。
- `block_tables` 每行为 `randperm(source_blocks)`（`source_blocks = ceil(source_len/block_size)`），int32；输出为紧凑 `[total_tokens, ...]`，因此没有 hbm_slots / copy_cap 参数。
- 空列表、重复值、负数、超出可见数量的卡数、不满足 `0 <= copy_count <= source_len` 的 copy 数，在启动 worker 前报错。

## 单组与单 die 入口

也可直接调用 Python launcher，`--cards N` 与 `--devices` 二选一；后者是可见列表内重编号后的逻辑索引。省略 `--output` 时，按 cards/BS/copy 自动命名并写入默认结果目录：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 tests/test_gather_pa_kv_cache_multiprocess.py \
  --cards 4 --batch-size 24 --copy-count 300 \
  --source-len 65536 --warmup 10 --iters 1000 --same-seed --seed 7 \
  --output results/manual/cards4_bs24_copy300.json

python3 tests/test_gather_pa_kv_cache.py \
  --device npu:0 --batch-size 24 --copy-count 300 \
  --source-len 65536 --warmup 10 --iters 1000
```

`--copy-count` 同时设置 `--copy-min/--copy-max`。随机范围参数仍可用于正确性测试；分析这种结果时 `copy_count` 显示为 `min..max`，不会把上限冒充固定 copy 数。

## 访问模式：block 粒度（默认）与 token 粒度（--block-size 1）

算子语义是"每请求 gather 逻辑区间 `0..seq_len-1`"，物理位置 = `block_tables[t // block_size] * block_size + t % block_size`。因此访问模式由 `--block-size` 决定：

- **`--block-size 128`（默认，paged attention 布局）**：copy=N 是每请求一段"N 个 token 的前缀"，块内连续（128 token ≈ 82KB）、块间随机跳——对应 chunked prefill / 整段前缀加载场景。
- **`--block-size 1`（token 粒度）**：每个 block 表项直接是一个 token 的物理位置，copy=N 就是 N 个**随机散落 token** 的 656B 行 gather——对应 DSA offload 稀疏 miss 场景，与 scatter_copy C8 的访问模式同口径。注意：这只是用算子表达出了该负载；CANN 原始 kernel 按 block 组织拷贝且每次拷贝同步等待，block=1 时效率预期显著低于 block=128，这正是要测的。

非默认 block_size 的 case 文件名带 `_blk<N>` 后缀（如 `cards1_bs8_copy100_blk1.json`）；不同 block_size 的结果混在同一目录分析时会因 cards/BS/copy 重复而报错，请用 `RESULTS_DIR` 分开存放。

单卡 D2D token 粒度示例（一张卡、每请求 300 个随机 token、HBM 源）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python3 tests/test_gather_pa_kv_cache.py \
  --device npu:0 --source-memory hbm --block-size 1 \
  --batch-size 24 --copy-count 300 --warmup 10 --iters 1000
```

整机扫参同理，加环境变量即可，例如 D2D + token 粒度：

```bash
SOURCE_MEMORY=hbm BLOCK_SIZE=1 RESULTS_DIR=results/gather_pa_kv_cache_d2d_blk1 \
A5_TEST_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CARD_COUNTS="1 2 4 8" BATCH_SIZES="8 16 24 32" \
  bash tests/run_gather_pa_kv_cache_multiprocess.sh
```

每组校验内容：CPU 参考 gather（经 `block_tables` 逐 token 映射到源物理行）与设备输出**按位比较**；每次校验前对输出写 poison；first-fill 容量校验（单请求 `seq_len=source_len`）；`copy_count=0` 时 `seq_lens` 全零、输出为空张量，调用必须是 no-op；输入只读校验（`block_tables`/`seq_lens`/`seq_offset` 调用前后逐字节不变；`--source-memory hbm` 时源 cache 也直接回读校验，DRAM/swapped-memory 源不做 D2H 回读，其字节完好性由输出的按位比较间接证明）；caller-owned 校验（调用前后 `data_ptr` 不变；该 torch_npu 封装可能无返回值，只打印实际返回类型，不断言）。

## 六列结果与带宽口径

终端结果表、CSV 和 Markdown 均只输出以下六列，顺序固定，时间/带宽保留三位小数：

```text
cards  batch_size  copy_count  avg_us_mean  avg_us_max  avg_bandwidth
```

对每个 die，先用 NPU Event 包围 `iters` 次 Gather，将计时时间除以 `iters`，得到 `t_i`（μs）。`D_i` 是该 die **每轮实际有效搬移字节数**。定义：

```text
avg_us_mean  = mean(t_i)
avg_us_max   = max(t_i)
avg_bandwidth = sum(D_i / (t_i * 1000))   # GB/s，先逐 die 计算再求和
```

`avg_bandwidth` 虽含 avg，含义是基于各 die 平均耗时计算的**带宽之和**，不再除以 cards。只计算一次有效 payload，不把源读取与目标写入重复计数，不使用整块分配容量。默认 `D_i = gathered_tokens_i × 656`（512B FP8 latent + 128B RoPE + 16B scale，与 C8 scatter_copy 口径一致）；覆盖 head size 时按实际配置计算。零 copy 的带宽为 0。

`avg_us_max` 是各 die 平均耗时的最大值，不是某次调用的最大耗时。固定 metadata 会重复 gather；初始化、正确性校验及 warmup 不在计时区间。该带宽是此并发测试的有效吞吐估计，结果不代表总线硬件计数器读数。

## 结果文件

默认输出目录为 `results/gather_pa_kv_cache`。每组生成 JSON 和日志，并生成统一命名的汇总文件：

- `gather_pa_kv_cache_timing_summary.csv`：六列数据。
- `gather_pa_kv_cache_timing_summary.md`：相同六列 Markdown 表。
- `gather_pa_kv_cache_multiprocess_summary.json`：同口径、未舍入的结果行。
- `gather_pa_kv_cache_run_manifest.json`：本轮 case 清单及完成状态。每次扫参只汇总本轮选择的 case；失败的重跑不会继续使用上次成功结果。

逐 case JSON 保存配置、每 die 耗时、有效字节数和正确性诊断，便于复核；worker 的完整日志保存在对应 `.log`，失败时终端显示日志末尾。

重新分析已有结果（无需 NPU，全部只依赖 Python 标准库）：

```bash
python3 tests/analyze_gather_pa_kv_cache_results.py \
  --results-dir results/gather_pa_kv_cache
python3 tests/format_gather_pa_kv_cache_csv.py \
  --input results/gather_pa_kv_cache/gather_pa_kv_cache_timing_summary.csv
```

没有本轮清单的目录会读取所有 `cards*.json`；若同一组 cards/BS/copy 出现多份结果，会报错并要求分开分析，避免静默合并不同配置。

## CPU 工具回归测试

```bash
python3 tests/test_gather_tools.py -v
```

该检查覆盖参数传递、工作负载边界校验、逐 die 带宽求和、零 copy、有效字节数校验、六列表格和扫参 dry-run 计划生成；它不替代目标 Ascend 950 上的算子正确性与带宽实测。

## 注意事项

- 源 KV cache 默认用 `torch_npu.empty_with_swapped_memory` 分配（真实主机 DRAM）。若算子拒绝 swapped-memory 源，加 `--source-memory hbm`（或 `SOURCE_MEMORY=hbm`）降级为普通 HBM 源做测量，并在结果解读时注明。
- fp8 布局需要支持 `float8_e4m3fn` 的 torch_npu；fp8 张量经 int8 存储初始化后 `.view(torch.float8_e4m3fn)` 得到，比较时按 int8/int16 按位比。
- `seq_lens` 按 vLLM-ascend 用法是 **per-request 长度**（非 cumsum）；`seq_offset` 固定全零。
- 可选 NUMA 绑定：`GATHER_PA_NUMA_NODES="0,1,..."`（每设备一个 node id，需 `numactl` 在 PATH 上）；默认不绑定。
- 若 `copy_count=0` 时空输出导致算子报错，worker 会如实失败；请以实际硬件表现为准。
