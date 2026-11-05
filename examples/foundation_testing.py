"""Zero-dependency Python unittest lab.

Run from repository root:
    python examples/foundation_testing.py

The suite verifies expected outputs, invalid inputs, exception types, file
isolation and a deterministic fake tool. No external API calls are made.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from beginner_tool_agent import run_request


def success_rate(completed, total):
    if type(completed) is not int or type(total) is not int:
        raise TypeError("counts must be integers")
    if total <= 0 or not 0 <= completed <= total:
        raise ValueError("invalid task counts")
    return completed / total


class FakeTool:
    """An external dependency that fails deterministically on its first call."""

    def __init__(self):
        self.calls = 0

    def execute(self):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("synthetic timeout")
        return {"ok": True}


def retry_once_safe(fake):
    """Demonstrate retries ONLY for a caller-provided idempotent fake tool."""
    try:
        return fake.execute()
    except TimeoutError:
        return fake.execute()


class SuccessRateTests(unittest.TestCase):
    def test_example(self):
        self.assertAlmostEqual(success_rate(7, 10), 0.7)

    def test_all_fail(self):
        self.assertEqual(success_rate(0, 8), 0.0)

    def test_all_pass(self):
        self.assertEqual(success_rate(8, 8), 1.0)

    def test_total_zero_rejected(self):
        with self.assertRaises(ValueError):
            success_rate(0, 0)

    def test_negative_count_rejected(self):
        with self.assertRaises(ValueError):
            success_rate(-1, 8)

    def test_more_successes_than_total_rejected(self):
        with self.assertRaises(ValueError):
            success_rate(9, 8)

    def test_boolean_not_accepted_as_integer(self):
        with self.assertRaises(TypeError):
            success_rate(True, 8)

    def test_text_not_accepted_as_integer(self):
        with self.assertRaises(TypeError):
            success_rate("7", 10)


class ToolAndSystemTests(unittest.TestCase):
    def test_json_tool_success(self):
        request = {
            "task_id": "case-001", "tool": "add",
            "arguments": {"a": 7, "b": 11},
        }
        response = run_request(json.dumps(request))
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"], 18)

    def test_unknown_tool_fails(self):
        request = {
            "task_id": "case-002", "tool": "exec_python",
            "arguments": {"a": 7, "b": 11},
        }
        response = run_request(json.dumps(request))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "ValueError")

    def test_input_error_precedes_execution(self):
        request = {
            "task_id": "case-003", "tool": "add",
            "arguments": {"a": True, "b": 11},
        }
        response = run_request(json.dumps(request))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "TypeError")

    def test_fake_timeout_recovery(self):
        fake = FakeTool()
        self.assertEqual(retry_once_safe(fake), {"ok": True})
        self.assertEqual(fake.calls, 2)

    def test_output_isolated_to_temporary_directory(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "result.txt"
            p.write_text("PASS", encoding="utf-8")
            self.assertEqual(p.read_text(encoding="utf-8"), "PASS")
            self.assertTrue(p.exists())
        self.assertFalse(p.exists())

    def test_python_cli_as_separate_process(self):
        script = Path(__file__).with_name("beginner_cli.py")
        process = subprocess.run(
            [sys.executable, str(script), "add", "12", "30"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(process.returncode, 0)
        self.assertEqual(process.stdout.strip(), "42")

    def test_bad_cli_input_returns_nonzero(self):
        script = Path(__file__).with_name("beginner_cli.py")
        process = subprocess.run(
            [sys.executable, str(script), "add", "12", "not-an-integer"],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("invalid int value", process.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
