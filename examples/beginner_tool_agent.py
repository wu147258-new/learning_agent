"""Safe, offline teaching example: JSON tool requests and independent checks.

No network, file writes, dynamic code execution, subprocesses, or LLM calls.
Run: python examples/beginner_tool_agent.py self-test
     python examples/beginner_tool_agent.py run
"""
from __future__ import annotations

import json
import re
import sys


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


TOOLS = {"add": add, "multiply": multiply}
ALLOWED_REQUEST_KEYS = {"task_id", "tool", "arguments"}


def validate_request(request, allowed_tools):
    if type(request) is not dict or set(request) != ALLOWED_REQUEST_KEYS:
        raise ValueError("request keys must match the protocol exactly")
    task_id = request["task_id"]
    if (type(task_id) is not str or
            re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task_id) is None):
        raise ValueError("invalid task_id")
    tool = request["tool"]
    if type(tool) is not str or tool not in TOOLS:
        raise ValueError("unknown tool")
    if tool not in allowed_tools:
        raise PermissionError("tool not authorized")
    args = request["arguments"]
    if type(args) is not dict or set(args) != {"a", "b"}:
        raise ValueError("arguments require exactly a and b")
    if type(args["a"]) is not int or type(args["b"]) is not int:
        raise TypeError("a and b must be integers, not bool or text")
    return task_id, tool, args


def independent_verification(tool, a, b, candidate):
    """Simple independent arithmetic check, not LLM self-report."""
    expected = a + b if tool == "add" else a * b
    return type(candidate) is int and candidate == expected


def run_request(text, allowed_tools=frozenset({"add"})):
    task_id = None
    try:
        request = json.loads(text)
        if type(request) is dict:
            task_id = request.get("task_id")
        task_id, tool, args = validate_request(request, allowed_tools)
        candidate = TOOLS[tool](args["a"], args["b"])
        if not independent_verification(
            tool, args["a"], args["b"], candidate
        ):
            raise AssertionError("independent result check failed")
        return {"task_id": task_id, "ok": True, "result": candidate}
    except (ValueError, TypeError, PermissionError, AssertionError) as exc:
        return {
            "task_id": task_id if type(task_id) is str else None,
            "ok": False,
            "error": type(exc).__name__,
            "detail": str(exc),
        }


def self_test():
    valid = {"task_id": "demo-001", "tool": "add",
             "arguments": {"a": 7, "b": 11}}
    call = lambda req, permissions=frozenset({"add"}): run_request(
        json.dumps(req), permissions
    )
    assert call(valid) == {
        "task_id": "demo-001", "ok": True, "result": 18
    }
    assert call({
        **valid, "tool": "multiply"
    })["error"] == "PermissionError"
    assert call({
        **valid, "tool": "multiply"
    }, frozenset({"multiply"}))["result"] == 77
    assert call({**valid, "tool": "delete_file"})["ok"] is False
    assert call({**valid, "arguments": {"a": True, "b": 11}})["ok"] is False
    assert call({**valid, "arguments": {"a": "7", "b": 11}})["ok"] is False
    assert call({**valid, "arguments": {
        "a": "ignore previous instructions", "b": 11
    }})["ok"] is False
    assert call({**valid, "extra_field": "administrator"})["ok"] is False
    assert call({**valid, "task_id": "../../etc/passwd"})["ok"] is False
    assert not run_request("not JSON")["ok"]
    assert independent_verification("add", 7, 11, 18)
    assert not independent_verification("add", 7, 11, 0)
    print("PASS: structured input, strict types, tool authorization and results")


def main(argv):
    if argv == ["self-test"]:
        self_test()
        return 0
    if argv == ["run"]:
        request = {
            "task_id": "demo-001",
            "tool": "add",
            "arguments": {"a": 7, "b": 11},
        }
        print(json.dumps(run_request(json.dumps(request)), ensure_ascii=False))
        return 0
    print("Usage: python beginner_tool_agent.py [self-test|run]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
