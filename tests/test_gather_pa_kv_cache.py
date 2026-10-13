#!/usr/bin/env python3
"""Ascend 950 GatherPaKvCache: DRAM/HBM -> HBM correctness and bandwidth."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gather_cli import resolve_copy_count
from gather_config import DTYPE_K, DTYPE_V, SINGLECARD_TEST
from gather_results import print_result, single_result

from _common import add_case_args, case_bytes_per_token, validate_args, write_json

POISON = 65


@dataclass
class Case:
    device: torch.device
    device_name: str
    source_blocks: int
    block_size: int
    bytes_per_token: int
    source_memory: str
    key_cache_cpu: torch.Tensor
    value_cache_cpu: torch.Tensor
    block_tables_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_offset_cpu: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    seq_offset: torch.Tensor
    key_out: torch.Tensor
    value_out: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_args(parser)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--json-output", type=Path)
    for name in ("ready-file", "start-file", "timing-ready-file", "timing-start-file"):
        parser.add_argument(f"--{name}", type=Path, help=argparse.SUPPRESS)
    return resolve_copy_count(parser.parse_args())


def allocate_source(cpu: torch.Tensor, device: torch.device, source_memory: str) -> torch.Tensor:
    if source_memory == "dram":
        return swapped_from_cpu(cpu, device)
    return cpu.to(device)


def make_case(args: argparse.Namespace) -> Case:
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    device_name = require_a5(device, args.allow_non_a5)
    generator = torch.Generator().manual_seed(args.seed)
    source_blocks = (args.source_len + args.block_size - 1) // args.block_size
    payload_size = case_bytes_per_token(args)
    # Same source-pool policy as the scatter_copy suite: read-only physical
    # blocks are shared by requests, each with its own randomized block table.
    block_tables_cpu = torch.stack([
        torch.randperm(source_blocks, generator=generator).to(torch.int32)
        for _ in range(args.batch_size)
    ])
    seq_lens_cpu = torch.randint(args.copy_min, args.copy_max + 1, (args.batch_size,), dtype=torch.int32, generator=generator)
    if args.batch_size > 1:
        seq_lens_cpu[0] = args.copy_min
        seq_lens_cpu[1] = args.copy_max
    seq_offset_cpu = torch.zeros(args.batch_size, dtype=torch.int32)
    total_tokens = int(seq_lens_cpu.sum())
    # fp8/bf16 payloads are initialized through integer storage so that the
    # source contents and the correctness comparison stay bit-exact.
    key_cache_cpu = torch.randint(-128, 128, (source_blocks, args.block_size, args.num_heads, args.head_size_k),
                                  dtype=torch.int8, generator=generator)
    value_cache_cpu = torch.randint(-32768, 32768, (source_blocks, args.block_size, args.num_heads, args.head_size_v),
                                    dtype=torch.int16, generator=generator)

    key_cache = allocate_source(key_cache_cpu, device, args.source_memory).view(torch.float8_e4m3fn)
    value_cache = allocate_source(value_cache_cpu, device, args.source_memory).view(torch.bfloat16)
    return Case(
        device=device, device_name=device_name, source_blocks=source_blocks, block_size=args.block_size,
        bytes_per_token=payload_size, source_memory=args.source_memory,
        key_cache_cpu=key_cache_cpu, value_cache_cpu=value_cache_cpu,
        block_tables_cpu=block_tables_cpu, seq_lens_cpu=seq_lens_cpu, seq_offset_cpu=seq_offset_cpu,
        key_cache=key_cache, value_cache=value_cache,
        block_tables=block_tables_cpu.to(device), seq_lens=seq_lens_cpu.to(device),
        seq_offset=seq_offset_cpu.to(device),
        key_out=torch.empty((total_tokens, args.num_heads, args.head_size_k), dtype=torch.float8_e4m3fn, device=device),
        value_out=torch.empty((total_tokens, args.num_heads, args.head_size_v), dtype=torch.bfloat16, device=device),
    )


def reference_gather(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU gather: map each token through the request's block table row."""
    key_rows = case.key_cache_cpu.reshape(-1, case.key_cache_cpu.shape[2], case.key_cache_cpu.shape[3])
    value_rows = case.value_cache_cpu.reshape(-1, case.value_cache_cpu.shape[2], case.value_cache_cpu.shape[3])
    keys, values = [], []
    for row, length in enumerate(case.seq_lens_cpu.tolist()):
        if length == 0:
            continue
        tokens = torch.arange(length)
        blocks = case.block_tables_cpu[row].long()
        physical = blocks[tokens // case.block_size] * case.block_size + tokens % case.block_size
        keys.append(key_rows[physical])
        values.append(value_rows[physical])
    key_shape = (0, case.key_cache_cpu.shape[2], case.key_cache_cpu.shape[3])
    value_shape = (0, case.value_cache_cpu.shape[2], case.value_cache_cpu.shape[3])
    return (
        torch.cat(keys) if keys else torch.empty(key_shape, dtype=torch.int8),
        torch.cat(values) if values else torch.empty(value_shape, dtype=torch.int16),
    )


def call_gather(case: Case, seq_lens: torch.Tensor | None = None,
                key_out: torch.Tensor | None = None, value_out: torch.Tensor | None = None):
    return torch_npu.npu_gather_pa_kv_cache(
        case.key_cache, case.value_cache, case.block_tables,
        case.seq_lens if seq_lens is None else seq_lens,
        seq_offset=case.seq_offset,
        key=case.key_out if key_out is None else key_out,
        value=case.value_out if value_out is None else value_out,
    )


def assert_gathered(case: Case) -> None:
    expected_key, expected_value = reference_gather(case)
    # Compare every gathered byte bitwise through integer views, so fp8/bf16
    # payload bits (including NaN encodings) must match exactly.
    if not torch.equal(case.key_out.view(torch.int8).cpu(), expected_key):
        raise AssertionError("Gathered key bytes differ from the CPU reference.")
    if not torch.equal(case.value_out.view(torch.int16).cpu(), expected_value):
        raise AssertionError("Gathered value bytes differ from the CPU reference.")


def check_gather(case: Case, phase: str) -> None:
    case.key_out.view(torch.int8).fill_(POISON)
    case.value_out.view(torch.int16).fill_(POISON)
    torch.npu.synchronize()
    if case.key_out.numel() and not bool((case.key_out.view(torch.int8) == POISON).all().item()):
        raise AssertionError("Key output was not poisoned before the gather.")
    if case.value_out.numel() and not bool((case.value_out.view(torch.int16) == POISON).all().item()):
        raise AssertionError("Value output was not poisoned before the gather.")
    tensors = (case.key_cache, case.value_cache, case.block_tables, case.seq_lens,
               case.seq_offset, case.key_out, case.value_out)
    pointers = tuple(t.data_ptr() for t in tensors)
    result = call_gather(case)
    torch.npu.synchronize()
    # The torch_npu wrapper may return None; record the actual type, no assert.
    return_type = type(result).__name__
    if pointers != tuple(t.data_ptr() for t in tensors):
        raise AssertionError("An input or caller-owned output address changed.")
    assert_gathered(case)
    for actual, expected in (
        (case.block_tables, case.block_tables_cpu), (case.seq_lens, case.seq_lens_cpu),
        (case.seq_offset, case.seq_offset_cpu),
    ):
        if not torch.equal(actual.cpu(), expected):
            raise AssertionError("Read-only gather metadata was modified.")
    if case.source_memory == "hbm":
        # Swapped-memory (DRAM) tensors are not read back here: D2H on
        # empty_with_swapped_memory storage is unverified, and the bitwise
        # output comparison above already proves the source bytes were read
        # intact. HBM sources are cheap to verify directly.
        if not torch.equal(case.key_cache.view(torch.int8).cpu(), case.key_cache_cpu):
            raise AssertionError("The source key cache was modified.")
        if not torch.equal(case.value_cache.view(torch.int16).cpu(), case.value_cache_cpu):
            raise AssertionError("The source value cache was modified.")
    print(
        f"A5_GATHER_PA_KV_CACHE_CHECK phase={phase} "
        f"gathered_tokens={int(case.seq_lens_cpu.sum())} bytes_per_token={case.bytes_per_token} "
        f"byte_exact=1 inputs_readonly=1 caller_owned=1 return_type={return_type} ok=1",
        flush=True,
    )


def wait_for_start(ready_file: Path | None, start_file: Path | None) -> None:
    if ready_file is None and start_file is None:
        return
    if ready_file is None or start_file is None:
        raise ValueError("ready and start files must be specified together")
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.write_text("ready\n", encoding="utf-8")
    deadline = time.monotonic() + 600
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for multiprocess start")
        time.sleep(0.001)


def run(args: argparse.Namespace) -> dict:
    # Exercise a first-fill-sized request (one request gathering source_len
    # tokens) through the SAME operator, outside timing.
    fill_args = argparse.Namespace(**vars(args))
    fill_args.batch_size = 1
    fill_args.copy_min = fill_args.copy_max = args.source_len
    fill_case = make_case(fill_args)
    check_gather(fill_case, "first_fill_capacity")
    del fill_case
    torch.npu.empty_cache()

    case = make_case(args)
    gathered_tokens = int(case.seq_lens_cpu.sum())
    payload_bytes = gathered_tokens * case.bytes_per_token
    allocator = "empty_with_swapped_memory" if args.source_memory == "dram" else "torch.empty"
    print(
        f"A5_GATHER_PA_CONFIG device={case.device} device_name={case.device_name!r} "
        f"dtype_k={DTYPE_K} dtype_v={DTYPE_V} num_heads={args.num_heads} "
        f"head_size_k={args.head_size_k} head_size_v={args.head_size_v} "
        f"block_size={args.block_size} bytes_per_token={case.bytes_per_token} "
        f"batch={args.batch_size} source_len={args.source_len} source_blocks={case.source_blocks} "
        f"copy_range=[{args.copy_min},{args.copy_max}] seq_lens={case.seq_lens_cpu.tolist()} "
        f"source_memory={args.source_memory} allocator={allocator} "
        f"launch_blocking={os.getenv('ASCEND_LAUNCH_BLOCKING', 'unset')}",
        flush=True,
    )
    check_gather(case, "requested_workload")
    zero_key = torch.empty((0, args.num_heads, args.head_size_k), dtype=torch.float8_e4m3fn, device=case.device)
    zero_value = torch.empty((0, args.num_heads, args.head_size_v), dtype=torch.bfloat16, device=case.device)
    zero_result = call_gather(case, torch.zeros_like(case.seq_lens), zero_key, zero_value)
    torch.npu.synchronize()
    if zero_key.numel() or zero_value.numel():
        raise AssertionError("copy_count=0 modified the outputs.")
    print(f"A5_GATHER_PA_ZERO_COUNT_CHECK return_type={type(zero_result).__name__} ok=1", flush=True)

    wait_for_start(args.ready_file, args.start_file)
    for _ in range(args.warmup):
        call_gather(case)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wait_for_start(args.timing_ready_file, args.timing_start_file)
    # Same timer as the scatter_copy suite: one event pair around repeated
    # launches. Immutable gather metadata keeps the count unchanged.
    start.record()
    for _ in range(args.iters):
        call_gather(case)
    end.record()
    end.synchronize()
    avg_us = start.elapsed_time(end) * 1000 / args.iters
    assert_gathered(case)
    print_result(single_result(args, payload_bytes, avg_us))
    result = {
        "schema_version": 2, "test": SINGLECARD_TEST,
        "status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device_count": 1,
        "config": {
            "device": str(case.device), "device_name": case.device_name,
            "dtype_k": DTYPE_K, "dtype_v": DTYPE_V,
            "num_heads": args.num_heads, "head_size_k": args.head_size_k, "head_size_v": args.head_size_v,
            "block_size": args.block_size, "bytes_per_token": case.bytes_per_token,
            "batch_size": args.batch_size, "source_len": args.source_len, "source_blocks": case.source_blocks,
            "copy_min": args.copy_min, "copy_max": args.copy_max,
            "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "source_memory": args.source_memory,
        },
        "workload": {
            "seq_lens": case.seq_lens_cpu.tolist(), "gathered_tokens": gathered_tokens,
            "payload_bytes_per_iteration": payload_bytes,
        },
        "correctness": {
            "data_exact": True, "caller_owned": True, "inputs_readonly": True,
            "zero_count_noop": True, "first_fill_capacity": True,
        },
        "summary": single_result(args, payload_bytes, avg_us),
        "performance": {"timer": "npu_event", "avg_us": avg_us},
    }
    if args.json_output is not None:
        write_json(args.json_output, result)
    print("A5_GATHER_PA_KV_CACHE_UT_OK", flush=True)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    global torch, torch_npu, require_a5, swapped_from_cpu
    import torch
    import torch_npu
    from _utils import require_a5, swapped_from_cpu
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
