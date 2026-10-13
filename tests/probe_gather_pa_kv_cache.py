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
    return key_cache, value_cache, block_tables, seq_lens.to(device), (
        seq_offset.to(device) if seq_offset is not None else None), key, value


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
        message = str(error).replace("\n", " ")[:2000]
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

    # Scale bisect: the bench worker's first-fill case (batch=1, seq=source_len,
    # block=1, rows 512B/144B, randperm block table, no seq_offset) fails while
    # the tiny probe above passes. Sweep one knob at a time to find the trigger.
    print("=" * 60)
    print("row-size bisect (block=1, seq=4, num_blocks=4):")
    for row_k, row_v in ((8, 8), (512, 512), (144, 144), (512, 144), (576, 512), (64, 64), (128, 128), (256, 256)):
        run_scale(f"row{row_k}x{row_v}", num_blocks=4, block=1, seq=4, row_k=row_k, row_v=row_v)
    print("seq bisect (block=1, rows 512x144):")
    for seq in (4, 64, 256, 1024, 4096, 16384, 65536):
        run_scale(f"blk1_seq{seq}", num_blocks=seq, block=1, seq=seq)
    run_scale("blk1_seq4_wide_table", num_blocks=65536, block=1, seq=4)
    run_scale("blk128_seq65536", num_blocks=512, block=128, seq=65536)


def run_scale(name: str, num_blocks: int, block: int, seq: int,
              row_k: int = 512, row_v: int = 144) -> None:
    try:
        device = "npu:0"
        generator = torch.Generator().manual_seed(7)
        key_cpu = torch.randint(-128, 128, (num_blocks, block, 1, row_k), dtype=torch.int8, generator=generator)
        value_cpu = torch.randint(-128, 128, (num_blocks, block, 1, row_v), dtype=torch.int8, generator=generator)
        perm_cpu = torch.randperm(num_blocks, generator=generator).to(torch.int32)
        seq_lens = torch.tensor([seq], dtype=torch.int32)
        key = torch.full((seq, 1, row_k), 65, dtype=torch.int8, device=device)
        value = torch.full((seq, 1, row_v), 65, dtype=torch.int8, device=device)
        result = torch_npu.npu_gather_pa_kv_cache(
            key_cpu.to(device), value_cpu.to(device), perm_cpu.to(device), seq_lens.to(device),
            key=key, value=value)
        torch.npu.synchronize()
        # CPU reference: token t lives at physical row perm[t // block] * block + t % block
        tokens = torch.arange(seq)
        physical = perm_cpu.long()[tokens // block] * block + tokens % block
        want_key = key_cpu.reshape(-1, 1, row_k)[physical]
        want_value = value_cpu.reshape(-1, 1, row_v)[physical]
        got = key.cpu()
        key_match = torch.equal(got, want_key)
        value_match = torch.equal(value.cpu(), want_value)
        if key_match and value_match:
            status = "MATCH"
        elif bool((got == 65).all()):
            status = "NOTHING-WRITTEN"
        else:
            bad = int((got.reshape(seq, -1) != want_key.reshape(seq, -1)).any(dim=1).sum())
            status = f"WRITTEN-BUT-WRONG(bad_rows={bad}/{seq},value_match={value_match})"
        print(f"[{name}] result={status}, wrapper_return={type(result).__name__}")
        del key, value, key_cpu, value_cpu
        torch.npu.empty_cache()
    except Exception as error:  # noqa: BLE001 - probe reports everything
        message = str(error).replace("\n", " ")[:2000]
        print(f"[{name}] FAILED: {message}")


if __name__ == "__main__":
    main()
