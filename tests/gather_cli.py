"""CLI validation shared by the sweep and hardware workers (no Torch import)."""

from __future__ import annotations

import os
from pathlib import Path

from gather_config import HEAD_SIZE_K, HEAD_SIZE_V, NUM_HEADS, RESULTS_SUBDIR


def parse_ids(value: str) -> list[int]:
    parts = value.split(",")
    if any(not part.strip().isdigit() for part in parts):
        raise ValueError("Devices must be comma-separated non-negative integer indices.")
    ids = [int(part) for part in parts]
    if len(ids) != len(set(ids)):
        raise ValueError("Device indices must be unique.")
    return ids


def add_device_args(parser, default: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--devices", default=default, help="Logical die indices within ASCEND_RT_VISIBLE_DEVICES.")
    group.add_argument("--cards", type=int, help="Use the first N visible logical devices/dies.")


def selected_devices(args) -> list[int]:
    if args.cards is not None:
        if args.cards < 1:
            raise ValueError("--cards must be positive.")
        devices = list(range(args.cards))
    else:
        devices = parse_ids(args.devices)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is not None and max(devices) >= len(parse_ids(visible)):
        raise ValueError("Requested logical devices exceed ASCEND_RT_VISIBLE_DEVICES.")
    return devices


def add_copy_count_arg(parser) -> None:
    parser.add_argument("--copy-count", type=int, help="Fixed tokens gathered per request; sets both copy-min and copy-max.")


def resolve_copy_count(args):
    if args.copy_count is not None:
        args.copy_min = args.copy_max = args.copy_count
    if hasattr(args, "output") and args.output is None:
        cards = args.cards if args.cards is not None else len(parse_ids(args.devices))
        copies = str(args.copy_min) if args.copy_min == args.copy_max else f"{args.copy_min}-{args.copy_max}"
        args.output = Path(__file__).resolve().parents[1] / RESULTS_SUBDIR / f"cards{cards}_bs{args.batch_size}_copy{copies}.json"
    return args


def validate_workload(args) -> None:
    if args.batch_size < 1 or args.source_len < 1:
        raise ValueError("batch-size and source-len must be positive.")
    num_heads = getattr(args, "num_heads", NUM_HEADS)
    head_size_k = getattr(args, "head_size_k", HEAD_SIZE_K)
    head_size_v = getattr(args, "head_size_v", HEAD_SIZE_V)
    if num_heads < 1 or head_size_k < 1 or head_size_v < 1:
        raise ValueError("num-heads, head-size-k and head-size-v must be positive.")
    if not 0 <= args.copy_min <= args.copy_max <= args.source_len:
        raise ValueError("Require 0 <= copy-min <= copy-max <= source-len.")
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive.")
