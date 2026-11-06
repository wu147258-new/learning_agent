"""An intentionally limited static code-audit Agent teaching pipeline.

Parses Python AST only. Never evaluates the scanned program, launches shell,
accesses networks, or applies patches without an explicit approval binding.
This is not a complete taint analyzer. Run: python3 examples/code_audit_agent_lab.py
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile
import textwrap


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    line: int
    column: int
    evidence: str
    limitation: str


@dataclass(frozen=True)
class ScanResult:
    source_sha: str
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class Approval:
    source_sha: str
    patch_sha: str
    approver: str
    allowed_path: str


class AuditError(Exception):
    pass


def digest(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def symbol(node):
    """Resolve a dotted identifier without executing code."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = symbol(node.value)
        return (base + "." + node.attr) if base else None
    return None


def literal_boolean(node, value: bool):
    return isinstance(node, ast.Constant) and node.value is value


def source_lines(source):
    return source.splitlines()


def source_is_dynamic(node, tainted: set[str]):
    """Over-approximate local expressions; not a formal data-flow engine."""
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Call):
        callee = symbol(node.func)
        if callee in {"input", "sys.stdin.readline"}:
            return True
        if callee in {"str", "int", "float", "bytes"}:
            return any(source_is_dynamic(x, tainted) for x in node.args)
        return any(source_is_dynamic(x, tainted) for x in node.args)
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return any(source_is_dynamic(x, tainted)
                   for x in ast.iter_child_nodes(node))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return any(source_is_dynamic(x, tainted)
                   for x in ast.iter_child_nodes(node))
    if isinstance(node, ast.Subscript):
        # Treat indexing a tainted collection as tainted.
        return source_is_dynamic(node.value, tainted)
    return any(source_is_dynamic(x, tainted)
               for x in ast.iter_child_nodes(node))


def scan_python(source: str) -> ScanResult:
    lines = source_lines(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AuditError(f"syntax error near line {error.lineno}") from error
    tainted: set[str] = set()
    findings = []
    # This deliberately walks in statement order only for basic assignments.
    # Control flow, imports and alias analysis are outside its guarantees.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            flagged = source_is_dynamic(node.value, tainted)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if flagged:
                        tainted.add(target.id)
                    else:
                        tainted.discard(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value:
                if source_is_dynamic(node.value, tainted):
                    tainted.add(node.target.id)
        elif isinstance(node, ast.Call):
            name = symbol(node.func)
            rule, sev, limitation = None, None, None
            if name in {"eval", "builtins.eval"}:
                rule, sev = "DYNAMIC_EVAL", "high"
                limitation = "AST flags dynamic eval, not exploitability."
            elif name in {"os.system", "subprocess.run",
                          "subprocess.Popen", "subprocess.call"}:
                is_shell = (
                    name == "os.system" or
                    any(k.arg == "shell" and literal_boolean(k.value, True)
                        for k in node.keywords)
                )
                if is_shell:
                    rule, sev = "SHELL_EXECUTION", "high"
                    limitation = "Shell execution may be intentional; review input control."
            elif name in {"pickle.loads", "pickle.load"}:
                rule, sev = "UNSAFE_DESERIALIZATION", "high"
                limitation = "Only unsafe when data trust is not established."
            elif name in {"yaml.load"}:
                has_safe_loader = any(
                    k.arg == "Loader" and symbol(k.value) in {
                        "yaml.SafeLoader", "SafeLoader"
                    }
                    for k in node.keywords
                )
                if not has_safe_loader:
                    rule, sev = "YAML_UNSAFE_LOADER", "medium"
                    limitation = "Check library version and loader semantics."
            elif name in {"cursor.execute", "db.execute"}:
                if node.args and isinstance(node.args[0],
                                            (ast.JoinedStr, ast.BinOp)):
                    rule, sev = "DYNAMIC_SQL", "high"
                    limitation = "Syntax heuristic; dataflow not fully resolved."
            if rule is not None:
                line = getattr(node, "lineno", 1)
                evidence = lines[line-1].strip() if line <= len(lines) else ""
                findings.append(Finding(
                    rule, sev, line, getattr(node, "col_offset", 0),
                    evidence, limitation
                ))
    # The AST walker order does not guarantee a precise data-flow solution.
    findings.sort(key=lambda f: (f.line, f.column, f.rule))
    return ScanResult(digest(source), tuple(findings))


def summarize(result: ScanResult):
    return {
        "source_sha": result.source_sha,
        "total": len(result.findings),
        "rules": sorted({f.rule for f in result.findings}),
        "evidence": [
            {"line": f.line, "rule": f.rule,
             "severity": f.severity, "text": f.evidence}
            for f in result.findings
        ],
    }


def request_approval(source: str, revised: str,
                     approver: str, relative_path: str):
    if not approver or approver == "model":
        raise AuditError("approval requires an external human identity")
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise AuditError("path scope violation")
    try:
        ast.parse(revised)
    except SyntaxError as error:
        raise AuditError("patch does not parse") from error
    return Approval(digest(source), digest(revised),
                    approver, relative_path)


def commit_patch(path: Path, root: Path, original: str,
                 revised: str, approval: Approval, tester):
    root_abs = root.resolve()
    path_abs = path.resolve()
    try:
        relative = path_abs.relative_to(root_abs).as_posix()
    except ValueError as error:
        raise AuditError("outside approved workspace") from error
    if relative != approval.allowed_path:
        raise AuditError("approved path mismatch")
    if (digest(original) != approval.source_sha or
            digest(revised) != approval.patch_sha):
        raise AuditError("approval not bound to content")
    if not path_abs.is_file():
        raise AuditError("target file missing")
    current = path_abs.read_text(encoding="utf-8")
    if digest(current) != approval.source_sha:
        raise AuditError("source changed since approval")
    ast.parse(revised)
    if not tester(revised):
        raise AuditError("independent test rejected patch")
    # Educational same-directory atomic replacement; no fsync guarantees.
    temp = path_abs.with_name(path_abs.name + ".candidate")
    temp.write_text(revised, encoding="utf-8")
    temp.replace(path_abs)
    return digest(path_abs.read_text(encoding="utf-8"))


def synthetic_source():
    return textwrap.dedent(
        """
        import os
        import pickle
        import sqlite3
        import subprocess

        def run_job(cursor):
            sql = input("query: ")
            cursor.execute(f"SELECT * FROM tasks WHERE k='{sql}'")
            subprocess.run(sql, shell=True)
            eval(sql)
            pickle.loads(b"synthetic-only")
        """
    ).strip() + "\n"


def safe_replacement():
    return textwrap.dedent(
        """
        import sqlite3

        def run_job(cursor, key: str):
            return cursor.execute(
                "SELECT * FROM tasks WHERE k=?", (key,)
            )
        """
    ).strip() + "\n"


def expect(error_type, fn):
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_finding_rules():
    code = synthetic_source()
    report = scan_python(code)
    rules = {f.rule for f in report.findings}
    assert {"DYNAMIC_SQL", "SHELL_EXECUTION", "DYNAMIC_EVAL",
            "UNSAFE_DESERIALIZATION"} == rules, rules
    assert all(f.line >= 1 and f.evidence for f in report.findings)
    assert len(report.source_sha) == 64
    assert scan_python(safe_replacement()).findings == ()
    expect(AuditError, lambda: scan_python("def broken("))
    print("PASS AST rules, source evidence and safe-parametrized example")


def test_approval_workflow():
    before = synthetic_source()
    after = safe_replacement()
    approval = request_approval(before, after, "human-reviewer",
                                "component.py")
    with tempfile.TemporaryDirectory(prefix="code-audit-") as tmp:
        root = Path(tmp)
        path = root / "component.py"
        path.write_text(before, encoding="utf-8")
        expect(AuditError, lambda: commit_patch(
            path, root, before, after, approval, lambda source: False
        ))
        assert path.read_text(encoding="utf-8") == before
        stale = before + "# changed\n"
        path.write_text(stale, encoding="utf-8")
        expect(AuditError, lambda: commit_patch(
            path, root, before, after, approval, lambda source: True
        ))
        path.write_text(before, encoding="utf-8")
        actual = commit_patch(
            path, root, before, after, approval,
            lambda source: scan_python(source).findings == ()
        )
        assert actual == digest(after)
        assert path.read_text(encoding="utf-8") == after
        expect(AuditError, lambda: request_approval(
            before, after, "model", "component.py"
        ))
        expect(AuditError, lambda: request_approval(
            before, after, "human", "../escape.py"
        ))
        print("PASS review binding, stale version and independent test gate")


if __name__ == "__main__":
    test_finding_rules()
    test_approval_workflow()
    print("NOTE: AST heuristics; not complete interprocedural taint analysis.")
