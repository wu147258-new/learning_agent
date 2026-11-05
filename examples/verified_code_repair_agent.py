"""A complete, deliberately tiny code-repair Agent workflow.

Python >=3.10, standard library only. Run:
    python examples/verified_code_repair_agent.py

The agent is a deterministic proposer, not an LLM. Only an AST-whitelisted
two-argument arithmetic function is ever executed; this is NOT a general
sandbox for untrusted Python. No external repositories, shell or network.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import os


BUG = "def add(a, b):\n    return a - b\n"
FIX = "def add(a, b):\n    return a + b\n"
CASES = ((2, 3, 5), (-5, 2, -3), (0, 0, 0), (101, -1, 100))


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def check_fixture_language(source: str) -> None:
    """Permit exactly: def add(a,b): return a +/- b.

    This deliberately restrictive grammar prevents arbitrary untrusted code
    execution in the teaching experiment. It is NOT a secure Python sandbox.
    """
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("fixture must contain one function")
    func = tree.body[0]
    args = func.args
    if (func.name != "add" or func.decorator_list or
            args.posonlyargs or args.kwonlyargs or args.defaults or
            args.kw_defaults or args.vararg or args.kwarg or
            [arg.arg for arg in args.args] != ["a", "b"] or
            len(func.body) != 1 or not isinstance(func.body[0], ast.Return)):
        raise ValueError("unsupported function signature/body")
    value = func.body[0].value
    if not isinstance(value, ast.BinOp) or type(value.op) not in (ast.Add, ast.Sub):
        raise ValueError("only addition or subtraction are permitted")
    if (not isinstance(value.left, ast.Name) or value.left.id != "a" or
            not isinstance(value.right, ast.Name) or value.right.id != "b"):
        raise ValueError("expression must use only a and b")
    if func.returns is not None:
        raise ValueError("no annotations in teaching fixture")
    if any(arg.annotation is not None for arg in args.args):
        raise ValueError("no annotations in teaching fixture")


def independent_tests(source: str) -> tuple[bool, tuple[str, ...]]:
    """Execute only after strict AST validation, with empty builtins."""
    check_fixture_language(source)
    namespace: dict[str, object] = {"__builtins__": {}}
    exec(compile(source, "verified_fixture.py", "exec"), namespace)
    function = namespace["add"]
    failures = tuple(
        f"add({a},{b}) produced {function(a, b)}, expected {expected}"
        for a, b, expected in CASES
        if function(a, b) != expected
    )
    return not failures, failures


@dataclass(frozen=True)
class Proposal:
    relative_path: str
    original_sha: str
    replacement: str
    operation_key: str


class ScopedWorkspace:
    def __init__(self, root: Path, *, permitted_files: frozenset[str]):
        self.root = root.resolve()
        self.permitted_files = permitted_files
        self.audit: list[tuple[str, str]] = []
        self.committed_operations: dict[str, tuple[str, str]] = {}
        self.write_count = 0

    def _path(self, name: str) -> Path:
        # Only exact, relative, preauthorized paths may be accessed.
        if name not in self.permitted_files:
            raise PermissionError("file path is outside the task scope")
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise PermissionError("path escaped the workspace")
        return path

    def observe(self, name: str) -> tuple[str, str]:
        path = self._path(name)
        content = path.read_text(encoding="utf-8")
        self.audit.append(("OBSERVE", name))
        return content, digest(content)

    def propose(self, name: str, operation_key: str) -> Proposal:
        current, sha = self.observe(name)
        check_fixture_language(current)
        if current != BUG:
            raise ValueError("this lesson only fixes the known fixture")
        candidate = current.replace("return a - b", "return a + b")
        check_fixture_language(candidate)
        self.audit.append(("PROPOSE", operation_key))
        return Proposal(name, sha, candidate, operation_key)

    def apply(self, proposal: Proposal, *, human_approved: bool) -> str:
        if not human_approved:
            raise PermissionError("no explicit human approval for write")
        check_fixture_language(proposal.replacement)
        # Idempotency key identifies one exact payload, not arbitrary writes.
        payload = digest(
            proposal.relative_path + "\0" + proposal.original_sha
            + "\0" + proposal.replacement
        )
        existing = self.committed_operations.get(proposal.operation_key)
        if existing is not None:
            old_payload, old_result = existing
            if old_payload != payload:
                raise ValueError("operation key reused with different patch")
            self.audit.append(("REPLAY", proposal.operation_key))
            return old_result

        path = self._path(proposal.relative_path)
        old = path.read_text(encoding="utf-8")
        if digest(old) != proposal.original_sha:
            raise RuntimeError("stale proposal; source has changed")

        # Explicitly stage a temporary file in the same scoped directory.
        staged = path.with_name(path.name + ".staged")
        if staged.exists():
            raise RuntimeError("unexpected staging file exists")
        try:
            staged.write_text(proposal.replacement, encoding="utf-8")
            # The verifier is an independent deterministic function, not
            # a model self-assessment. Do NOT execute unreviewed code.
            passed, errors = independent_tests(staged.read_text("utf-8"))
            if not passed:
                self.audit.append(("REJECT_TESTS", "; ".join(errors)))
                raise AssertionError("independent acceptance tests failed")
            # Recheck version before committing (single-process toy).
            if digest(path.read_text("utf-8")) != proposal.original_sha:
                raise RuntimeError("source changed before commit")
            os.replace(staged, path)
            self.write_count += 1
            result_sha = digest(proposal.replacement)
            self.committed_operations[proposal.operation_key] = (
                payload, result_sha
            )
            self.audit.append(("VERIFIED_COMMIT", result_sha))
            return result_sha
        finally:
            staged.unlink(missing_ok=True)


def self_test() -> None:
    assert independent_tests(BUG)[0] is False
    assert independent_tests(FIX)[0] is True
    for disallowed in (
        "import os\n" + FIX,
        "def add(a, b):\n    return __import__('os')\n",
        "def add(a, b):\n    return a * b\n",
    ):
        try:
            check_fixture_language(disallowed)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe fixture was accepted")

    with TemporaryDirectory(prefix="agent-repair-") as directory:
        root = Path(directory)
        fixture = root / "calculator.py"
        fixture.write_text(BUG, encoding="utf-8")
        workspace = ScopedWorkspace(root, permitted_files=frozenset({
            "calculator.py"
        }))
        proposal = workspace.propose("calculator.py", "operation-17")
        try:
            workspace.apply(proposal, human_approved=False)
        except PermissionError:
            pass
        else:
            raise AssertionError("write without approval was allowed")
        assert fixture.read_text("utf-8") == BUG
        assert workspace.write_count == 0

        # Reject a patch which is valid Python but does not fix the problem.
        not_fixed = Proposal(
            "calculator.py", proposal.original_sha, BUG, "operation-bad"
        )
        try:
            workspace.apply(not_fixed, human_approved=True)
        except AssertionError:
            pass
        else:
            raise AssertionError("failed tests were accepted")
        assert fixture.read_text("utf-8") == BUG
        assert workspace.write_count == 0
        assert not (root / "calculator.py.staged").exists()

        sha = workspace.apply(proposal, human_approved=True)
        assert sha == digest(FIX)
        assert fixture.read_text("utf-8") == FIX
        assert workspace.write_count == 1
        # Repeated operation with same payload returns same result.
        assert workspace.apply(proposal, human_approved=True) == sha
        assert workspace.write_count == 1
        try:
            workspace.apply(Proposal(
                "calculator.py", digest(BUG), BUG, "operation-17"
            ), human_approved=True)
        except ValueError:
            pass
        else:
            raise AssertionError("reused idempotency key accepted different patch")
        try:
            workspace.apply(Proposal(
                "calculator.py", digest(BUG), FIX, "fresh-operation"
            ), human_approved=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale source hash was accepted")
        for wrong_path in ("../secrets.txt", "/tmp/calculator.py"):
            try:
                workspace.observe(wrong_path)
            except PermissionError:
                pass
            else:
                raise AssertionError("out-of-scope read was accepted")

        assert any(kind == "REJECT_TESTS" for kind, _ in workspace.audit)
        assert any(kind == "VERIFIED_COMMIT" for kind, _ in workspace.audit)
        print("PASS code repair: propose -> approval -> test -> atomic replace")
        print("PASS code repair: stale hash, scope, replay and invalid AST rejected")
        print(f"Workspace write effects: {workspace.write_count}")
        print("NOTE: The allowed AST grammar is a tiny lesson, NOT sandboxing.")


if __name__ == "__main__":
    self_test()
