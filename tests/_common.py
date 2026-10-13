"""Pure-Python CLI and result helpers shared by the single/multi-die tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gather_cli import add_copy_count_arg
from gather_config import (
    BLOCK_SIZE,
    HEAD_SIZE_K,
    HEAD_SIZE_V,
    NUM_HEADS,
    bytes_per_token,
)

MAX_SOURCE_LEN = 1 << 18


def add_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--source-len", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help="Cache block size in tokens; 128 matches paged-attention layout, "
                             "1 gives token-granular random gather (DSA sparse-miss pattern).")
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--head-size-k", type=int, default=HEAD_SIZE_K)
    parser.add_argument("--head-size-v", type=int, default=HEAD_SIZE_V)
    add_copy_count_arg(parser)
    parser.add_argument("--copy-min", type=int, default=0)
    parser.add_argument("--copy-max", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-memory", choices=("dram", "hbm"), default="dram",
                        help="dram: source caches via torch_npu.empty_with_swapped_memory; "
                             "hbm: plain device tensors (fallback if the operator rejects swapped-memory sources).")
    parser.add_argument("--allow-non-a5", action="store_true", help="Portability debugging only; not A5 validation.")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.source_len <= MAX_SOURCE_LEN:
        raise ValueError("--source-len must be in [1,262144].")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive.")
    if args.num_heads < 1 or args.head_size_k < 1 or args.head_size_v < 1:
        raise ValueError("--num-heads, --head-size-k and --head-size-v must be positive.")
    if not 0 <= args.copy_min <= args.copy_max <= args.source_len:
        raise ValueError("Require 0 <= copy-min <= copy-max <= source-len.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")


def case_bytes_per_token(args: argparse.Namespace) -> int:
    return bytes_per_token(args.num_heads, args.head_size_k, args.head_size_v)


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
