#!/usr/bin/env python3
"""CPU-only regression tests for the CLI, workload validation and result files."""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gather_cli import parse_ids, selected_devices, validate_workload
from gather_config import (
    BYTES_PER_TOKEN,
    DTYPE_K,
    DTYPE_V,
    LEGACY_VISIBLE_ENV,
    MULTIPROCESS_TEST,
    SINGLECARD_TEST,
)
from gather_results import (
    COLUMNS,
    build_rows,
    copy_label,
    effective_bandwidth,
    format_table,
    normalize_case,
)
from run_gather_pa_kv_cache_sweep import cases, integer_list, parse_args

ROOT = Path(__file__).resolve().parents[1]


def fixture(latencies=(10.0, 30.0), batch=3, copies=7):
    config = dict(dtype_k=DTYPE_K, dtype_v=DTYPE_V, num_heads=1, head_size_k=512, head_size_v=72,
                  block_size=128, bytes_per_token=BYTES_PER_TOKEN, batch_size=batch,
                  source_len=65536, source_blocks=512, copy_min=copies, copy_max=copies,
                  warmup=0, iters=5, seed=7, source_memory="dram")
    workers = []
    for rank, latency in enumerate(latencies):
        workers.append({
            "status": "passed", "test": SINGLECARD_TEST,
            "config": {**config, "device": f"npu:{rank}"}, "rank": rank,
            "correctness": {"data_exact": True},
            "workload": {"seq_lens": [copies] * batch, "gathered_tokens": batch * copies,
                         "payload_bytes_per_iteration": batch * copies * BYTES_PER_TOKEN},
            "performance": {"avg_us": latency},
        })
    return {"status": "passed", "test": MULTIPROCESS_TEST, "device_count": len(workers),
            "config": config, "summary": {"all_correct": True}, "per_device": workers}


class CliTests(unittest.TestCase):
    def test_parse_ids(self):
        self.assertEqual(parse_ids("0,1,2"), [0, 1, 2])
        for bad in ("", "1,,2", "1,1", "a,b", "-1,2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ids(bad)

    def test_integer_list(self):
        self.assertEqual(integer_list("1 2 4", "CARD_COUNTS", 1), [1, 2, 4])
        self.assertEqual(integer_list("0 100", "COPY_COUNTS", 0), [0, 100])
        for bad, minimum in (("", 0), ("0", 1), ("1 1", 1), ("-1", 0), ("a b", 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                integer_list(bad, "X", minimum)

    def test_validate_workload_bounds(self):
        base = dict(batch_size=8, source_len=65536, copy_min=0, copy_max=2048, warmup=10, iters=1000)
        validate_workload(argparse.Namespace(**base))
        # copy_max == source_len is the legal first-fill boundary.
        validate_workload(argparse.Namespace(**{**base, "copy_min": 65536, "copy_max": 65536}))
        for patch_values in ({"batch_size": 0}, {"source_len": 0}, {"copy_min": -1},
                             {"copy_min": 5, "copy_max": 4}, {"copy_max": 65537},
                             {"warmup": -1}, {"iters": 0}, {"num_heads": 0}, {"head_size_k": 0}):
            with self.subTest(patch=patch_values), self.assertRaises(ValueError):
                validate_workload(argparse.Namespace(**{**base, **patch_values}))

    def test_legacy_visibility_alias_and_device_mapping(self):
        with patch.dict(os.environ, {LEGACY_VISIBLE_ENV: "2,5"}, clear=True):
            self.assertEqual(parse_args([]).visible_devices, "2,5")
        with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "3,7"}, clear=True):
            self.assertEqual(selected_devices(argparse.Namespace(cards=2, devices=None)), [0, 1])


class ResultsTests(unittest.TestCase):
    def test_effective_bandwidth(self):
        self.assertAlmostEqual(effective_bandwidth(656000, 10.0), 65.6)
        self.assertEqual(effective_bandwidth(0, 10.0), 0.0)
        for time in (0, -1, float("inf"), float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                effective_bandwidth(1000, time)

    def test_unequal_die_latencies_sum_individual_rates(self):
        result = normalize_case(fixture())
        # 21 tokens per die at 10/30 us: sum the two per-die rates.
        expected = 21 * BYTES_PER_TOKEN / 10000 + 21 * BYTES_PER_TOKEN / 30000
        self.assertAlmostEqual(result["avg_bandwidth"], expected)
        self.assertEqual(result["avg_us_mean"], 20)
        self.assertEqual(result["avg_us_max"], 30)
        self.assertNotAlmostEqual(result["avg_bandwidth"], 42 * BYTES_PER_TOKEN / 20000)
        self.assertEqual(tuple(result), COLUMNS)

    def test_zero_copy_and_single_die(self):
        self.assertEqual(normalize_case(fixture(copies=0))["avg_bandwidth"], 0)
        row = normalize_case(fixture(latencies=(20,)))
        self.assertEqual(row["cards"], 1)
        self.assertEqual(row["avg_us_mean"], row["avg_us_max"])
        self.assertAlmostEqual(row["avg_bandwidth"], 21 * BYTES_PER_TOKEN / 20000)

    def test_copy_label(self):
        self.assertEqual(copy_label({"copy_min": 7, "copy_max": 7}), 7)
        self.assertEqual(copy_label({"copy_min": 0, "copy_max": 7}), "0..7")
        with self.assertRaises(ValueError):
            copy_label({"copy_min": 8, "copy_max": 7})

    def test_random_count_range_keeps_per_die_bytes(self):
        case = fixture()
        case["config"]["copy_min"] = 0
        for worker in case["per_device"]:
            worker["config"]["copy_min"] = 0
        case["per_device"][0]["workload"] = {"seq_lens": [1, 0, 0], "gathered_tokens": 1,
                                             "payload_bytes_per_iteration": BYTES_PER_TOKEN}
        row = normalize_case(case)
        self.assertEqual(row["copy_count"], "0..7")
        self.assertAlmostEqual(row["avg_bandwidth"],
                               BYTES_PER_TOKEN / 10000 + 21 * BYTES_PER_TOKEN / 30000)

    def test_normalize_case_rejects_bad_cases(self):
        bad = []
        case = fixture(); case["status"] = "failed"; bad.append(case)
        case = fixture(); case["device_count"] = 3; bad.append(case)
        case = fixture(); case["summary"]["all_correct"] = False; bad.append(case)
        # Wrong bytes/token in the recorded worker config.
        case = fixture(); case["per_device"][0]["config"]["bytes_per_token"] = 655; bad.append(case)
        # Payload inconsistent with gathered tokens.
        case = fixture(); case["per_device"][0]["workload"]["payload_bytes_per_iteration"] += 1; bad.append(case)
        # Gathered tokens disagree with the fixed copy count.
        case = fixture(); case["per_device"][0]["workload"]["gathered_tokens"] += 1; \
            case["per_device"][0]["workload"]["payload_bytes_per_iteration"] += BYTES_PER_TOKEN; bad.append(case)
        case = fixture(); case["per_device"][0]["config"]["dtype_k"] = "fp8_e5m2"; bad.append(case)
        case = fixture(); case["per_device"][0]["correctness"]["data_exact"] = False; bad.append(case)
        case = fixture(); case["per_device"][0]["config"]["head_size_k"] = 256; bad.append(case)
        for case in bad:
            with self.subTest(case=case), self.assertRaises(ValueError):
                normalize_case(case)

    def test_six_columns_in_text_and_markdown(self):
        rows = build_rows([fixture(latencies=(12,), batch=17, copies=321), fixture(batch=5, copies=0)])
        self.assertEqual(format_table(rows).splitlines()[0].split(), list(COLUMNS))
        markdown = format_table(rows, markdown=True)
        self.assertEqual([part.strip() for part in markdown.splitlines()[0].strip("|").split("|")], list(COLUMNS))
        self.assertIn("321", markdown)
        self.assertIn("17", markdown)

    def test_duplicate_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_rows([fixture(), fixture()])


class SweepTests(unittest.TestCase):
    def test_dry_run_plan_from_environment(self):
        with patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "0,1,2,3", "CARD_COUNTS": "1 2",
                                    "BATCH_SIZES": "8", "COPY_COUNTS": "0 100"}, clear=True):
            args = parse_args(["--dry-run"])
        planned = list(cases(args))
        self.assertEqual(len(planned), 4)
        self.assertEqual(args.visible_devices, "0,1,2,3")
        name, output, command = planned[-1]
        self.assertEqual(name, "cards2_bs8_copy100")
        self.assertIn("test_gather_pa_kv_cache_multiprocess.py", command[1])
        self.assertIn("--same-seed", command)
        self.assertEqual(command[command.index("--copy-count") + 1], "100")
        self.assertEqual(command[command.index("--source-memory") + 1], "dram")

    def test_command_line_overrides_environment(self):
        with patch.dict(os.environ, {"CARD_COUNTS": "999", "BATCH_SIZES": "0", "COPY_COUNTS": "-1"}, clear=True):
            args = parse_args(["--visible-devices", "4,9", "--cards", "2", "--batch-size", "8", "16",
                               "--copy-count", "200", "300"])
        self.assertEqual((args.cards, args.batch_size, args.copy_count), ([2], [8, 16], [200, 300]))

    def test_invalid_requests_fail_before_launch(self):
        for env in ({"CARD_COUNTS": "3"}, {"CARD_COUNTS": "0"}, {"CARD_COUNTS": "1 1"},
                    {"BATCH_SIZES": "0"}, {"COPY_COUNTS": "-1"}, {"COPY_COUNTS": "65537"},
                    {"COPY_COUNTS": ""}, {"ITERS": "0"}, {"TEST_VISIBLE_DEVICES": "1,,2"},
                    {"TEST_VISIBLE_DEVICES": "1,1"}):
            with self.subTest(env=env), patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "0,1", **env}, clear=True):
                with self.assertRaises(ValueError):
                    parse_args([])

    def test_failed_rerun_invalidates_previous_success(self):
        import contextlib
        import io
        import json
        import subprocess

        from run_gather_pa_kv_cache_sweep import run

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = parse_args(["--visible-devices", "0", "--cards", "1", "--batch-size", "3",
                               "--copy-count", "7", "--results-dir", directory])
            output = next(cases(args))[1]
            output.write_text(json.dumps(fixture(latencies=(10,))))
            with patch("run_gather_pa_kv_cache_sweep.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 1)), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    run(args)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")
            manifest = json.loads((Path(directory) / "gather_pa_kv_cache_run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_launcher_and_worker_cli_defaults_without_torch(self):
        import importlib
        import sys

        launcher = importlib.import_module("test_gather_pa_kv_cache_multiprocess")
        with patch.object(sys, "argv", ["test", "--cards", "2", "--batch-size", "17", "--copy-count", "321"]):
            args = launcher.parse_args()
        self.assertEqual(args.output.name, "cards2_bs17_copy321.json")
        self.assertEqual((args.num_heads, args.head_size_k, args.head_size_v), (1, 512, 72))
        self.assertEqual(args.block_size, 128)
        worker = importlib.import_module("test_gather_pa_kv_cache")
        with patch.object(sys, "argv", ["test", "--copy-count", "200"]):
            args = worker.parse_args()
        self.assertEqual((args.copy_min, args.copy_max), (200, 200))
        self.assertEqual((args.batch_size, args.source_len, args.warmup, args.iters), (24, 65536, 10, 1000))
        self.assertEqual(args.source_memory, "dram")

    def test_block_size_token_granular_mode(self):
        with patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "0", "CARD_COUNTS": "1",
                                    "BATCH_SIZES": "8", "COPY_COUNTS": "100"}, clear=True):
            args = parse_args(["--dry-run", "--block-size", "1"])
            name, _, command = list(cases(args))[0]
        self.assertEqual(name, "cards1_bs8_copy100_blk1")
        self.assertEqual(command[command.index("--block-size") + 1], "1")
        with patch.dict(os.environ, {"TEST_VISIBLE_DEVICES": "0", "BLOCK_SIZE": "0"}, clear=True):
            with self.assertRaises(ValueError):
                parse_args([])


if __name__ == "__main__":
    unittest.main()
