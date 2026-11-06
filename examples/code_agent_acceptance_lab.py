"""An acceptance check bound to source digest, in an isolated temp tree."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import ast


def digest(data: bytes):
    return sha256(data).hexdigest()


def verify_add(code):
    tree=ast.parse(code)
    # Tiny expression-only test; not a general Python sandbox.
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    if len(functions)!=1 or functions[0].name!="add":
        raise ValueError("unexpected function")
    expression=functions[0].body[0].value
    if not isinstance(expression,ast.BinOp) or not isinstance(expression.op,ast.Add):
        raise ValueError("required addition absent")
    assert isinstance(expression.left,ast.Name)
    assert isinstance(expression.right,ast.Name)
    return True


def main():
    bad="def add(a,b):\n    return a-b\n"
    good="def add(a,b):\n    return a+b\n"
    with TemporaryDirectory() as d:
        path=Path(d)/"calc.py"
        path.write_text(bad)
        base=digest(path.read_bytes())
        try:
            verify_add(path.read_text())
        except ValueError:
            pass
        else:
            raise AssertionError("original defect not caught")
        assert verify_add(good)
        # Simulate independent edit invalidating original proposal.
        path.write_text("def add(a,b):\n    return a*b\n")
        assert digest(path.read_bytes()) != base
        assert path.read_text() != good
        # Only a fresh version-bound candidate may be committed.
        fresh=digest(path.read_bytes())
        if digest(path.read_bytes()) == fresh:
            path.write_text(good)
        assert verify_add(path.read_text())
    print("PASS source digest conflict and independent AST acceptance")
    print("NOTE: no untrusted Python executed; AST check not a sandbox.")


if __name__=="__main__":
    main()
