"""Operating systems and concurrency: deterministic offline exercises.

Run: python examples/foundation_concurrency.py
Uses no network, persistent files, third-party packages or shell=True.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
import subprocess
import sys


@dataclass
class VersionedRecord:
    value: str = "pending"
    version: int = 10
    lock: Lock = field(default_factory=Lock)

    def read(self):
        with self.lock:
            return self.value, self.version

    def compare_and_swap(self, expected_version, new_value):
        with self.lock:
            if self.version != expected_version:
                return False
            self.value = new_value
            self.version += 1
            return True


def check_process():
    process = subprocess.run(
        [sys.executable, "-c", "print(2 + 3)"],
        capture_output=True, text=True, timeout=3, check=False,
    )
    assert process.returncode == 0
    assert process.stdout.strip() == "5"
    try:
        subprocess.run(
            [sys.executable, "-c", "import time;time.sleep(2)"],
            capture_output=True, timeout=0.08, check=False,
        )
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("timeout was not detected")


def check_threads():
    counter = {"value": 0}
    lock = Lock()

    def increment_many():
        for _ in range(1000):
            with lock:
                counter["value"] += 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(increment_many) for _ in range(8)]
        for item in futures:
            item.result()
    assert counter["value"] == 8000


async def async_worker(name):
    await asyncio.sleep(0.001)
    return name + ":done"


def check_async():
    async def gather():
        return await asyncio.gather(
            async_worker("A"), async_worker("B")
        )
    assert asyncio.run(gather()) == ["A:done", "B:done"]


def check_versions():
    record = VersionedRecord()
    a_value, a_version = record.read()
    b_value, b_version = record.read()
    assert a_value == b_value == "pending"
    assert a_version == b_version == 10
    assert record.compare_and_swap(a_version, "done")
    assert not record.compare_and_swap(b_version, "failed")
    assert record.read() == ("done", 11)
    assert record.compare_and_swap(11, "archived")
    assert record.read() == ("archived", 12)


if __name__ == "__main__":
    check_process()
    check_threads()
    check_async()
    check_versions()
    print("PASS: subprocess, timeout, thread lock, asyncio and versioned state")
