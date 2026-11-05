"""Minimal, deterministic command line exercise: one operation per process.

Commands:
  python examples/beginner_cli.py add 12 30
  python examples/beginner_cli.py self-test
"""
from __future__ import annotations

import argparse
import sys


def add(a: int, b: int) -> int:
    if type(a) is not int or type(b) is not int:
        raise TypeError("both operands must be integers")
    return a + b


def run_self_test() -> None:
    assert add(12, 30) == 42
    assert add(-4, 7) == 3
    assert add(0, 0) == 0
    try:
        add(True, 3)
    except TypeError:
        pass
    else:
        raise AssertionError("bool must not pass strict integer validation")
    try:
        add("12", 30)
    except TypeError:
        pass
    else:
        raise AssertionError("string must not pass strict integer validation")
    print("PASS: integer arithmetic and strict type checks")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small CLI exercise")
    sub = parser.add_subparsers(dest="command", required=True)
    add_parser = sub.add_parser("add", help="add two integers")
    add_parser.add_argument("a", type=int)
    add_parser.add_argument("b", type=int)
    sub.add_parser("self-test", help="run deterministic tests")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        run_self_test()
        return 0
    if args.command == "add":
        print(add(args.a, args.b))
        return 0
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
