#!/usr/bin/env python3
"""Probe the local torch_npu.npu_gather_pa_kv_cache wrapper semantics.

Prints the wrapper signature/docstring and runs minimal unambiguous gathers
(2 blocks x 4 tokens, 8-byte rows, block table selects physical block 1) under
different seq_lens/seq_offset conventions, reporting which variant lands the
expected bytes in the caller-allocated outputs.

Usage: python3 tests/probe_gather_pa_kv_cache.py
"""

from __future__ import annotations

import inspect

import torch
import torch_npu

NUM_BLOCKS = 2
BLOCK = 4
ROW = 8
BATCH = 1
SEQ = 4


def make_inputs(seq_lens: torch.Tensor, seq_offset):
    device = "npu:0"
    key_cache = torch.arange(NUM_BLOCKS * BLOCK * ROW, dtype=torch.int32).to(torch.int8).reshape(
        NUM_BLOCKS, BLOCK, 1, ROW).to(device)
    value_cache = (100 + torch.arange(NUM_BLOCKS * BLOCK * ROW, dtype=torch.int32)).to(torch.int8).reshape(
        NUM_BLOCKS, BLOCK, 1, ROW).to(device)
    block_tables = torch.ones((BATCH, 1), dtype=torch.int32, device=device)  # physical block 1
    key = torch.full((SEQ, 1, ROW), 65, dtype=torch.int8, device=device)
    value = torch.full((SEQ, 1, ROW), 65, dtype=torch.int8, device=device)
    return key_cache, value_cache, block_tables, seq_lens.to(device), seq_offset, key, value


def expected():
    base = torch.arange(NUM_BLOCKS * BLOCK * ROW, dtype=torch.int32).to(torch.int8).reshape(NUM_BLOCKS, BLOCK, 1, ROW)
    return base[1].reshape(SEQ, 1, ROW)  # physical block 1


def run_variant(name: str, seq_lens: torch.Tensor, seq_offset) -> None:
    try:
        key_cache, value_cache, block_tables, seq_lens_dev, seq_offset, key, value = make_inputs(seq_lens, seq_offset)
        kwargs = {}
        if seq_offset is not None:
            kwargs["seq_offset"] = seq_offset
        result = torch_npu.npu_gather_pa_kv_cache(
            key_cache, value_cache, block_tables, seq_lens_dev, key=key, value=value, **kwargs)
        torch.npu.synchronize()
        got = key.cpu()
        want = expected()
        status = "MATCH" if torch.equal(got, want) else (
            "WRITTEN-BUT-WRONG" if not bool((got == 65).all()) else "NOTHING-WRITTEN")
        print(f"[{name}] ok, result={status}, wrapper_return={type(result).__name__}")
        if status == "WRITTEN-BUT-WRONG":
            print(f"  expected row0={want[0, 0].tolist()} got row0={got[0, 0].tolist()}")
    except Exception as error:  # noqa: BLE001 - probe reports everything
        message = str(error).replace("\n", " ")[:300]
        print(f"[{name}] FAILED: {message}")


def main() -> None:
    torch.npu.set_device("npu:0")
    torch.npu.config.allow_internal_format = False
    func = torch_npu.npu_gather_pa_kv_cache
    try:
        print("signature:", inspect.signature(func))
    except (TypeError, ValueError):
        print("signature: <unavailable>")
    doc = inspect.getdoc(func)
    print("doc:", (doc[:800] if doc else "<none>"))
    print("=" * 60)

    zero_off32 = torch.zeros(BATCH, dtype=torch.int32)
    run_variant("per-request int32 + seq_offset", torch.tensor([SEQ], dtype=torch.int32), zero_off32)
    run_variant("per-request int32, no seq_offset", torch.tensor([SEQ], dtype=torch.int32), None)
    run_variant("per-request int64 + seq_offset", torch.tensor([SEQ], dtype=torch.int64),
                torch.zeros(BATCH, dtype=torch.int64))
    run_variant("cumsum int32 [0,4] + seq_offset", torch.tensor([0, SEQ], dtype=torch.int32), zero_off32)
    run_variant("cumsum int64 [0,4] + seq_offset", torch.tensor([0, SEQ], dtype=torch.int64),
                torch.zeros(BATCH, dtype=torch.int64))


if __name__ == "__main__":
    main()
