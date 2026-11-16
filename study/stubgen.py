"""Generate exercise stubs from reference solutions (author tool).

Rules applied to a solution file:
  * module docstring, imports, constants, classes without methods: kept as-is
  * every public function/method body -> its docstring (if any) + `raise NotImplementedError`
  * top-level private helpers (`_name`) are removed - the learner designs their own
  * names listed in a module-level ``__stub_keep__ = [...]`` are kept verbatim
    (use for scaffolding the learner should NOT write); ``__stub_keep__`` itself is removed

Existing exercise files are never overwritten unless ``force=True`` - your work is safe.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .curriculum import Module


def _stub_body(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    body: list[ast.stmt] = []
    doc = ast.get_docstring(fn, clean=False)
    if doc is not None:
        body.append(ast.Expr(ast.Constant(doc)))
    body.append(ast.Raise(exc=ast.Name("NotImplementedError", ast.Load()), cause=None))
    if isinstance(fn, ast.AsyncFunctionDef) and _has_yield(fn):
        # keep it an async generator so `async for stub()` raises NotImplementedError (TODO)
        body.append(ast.Expr(ast.Yield(None)))
    fn.body = body


def _has_yield(fn: ast.AsyncFunctionDef) -> bool:
    stack: list[ast.AST] = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return False


def _keep_names(tree: ast.Module) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__stub_keep__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def make_stub(source: str) -> str:
    tree = ast.parse(source)
    keep = _keep_names(tree)
    new_body: list[ast.stmt] = []
    dropped: list[ast.stmt] = []
    originals = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__stub_keep__" for t in node.targets
        ):
            continue
        name = getattr(node, "name", None)
        if name in keep:
            new_body.append(node)
            continue
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.startswith("_") and not t.id.startswith("__") for t in targets):
                dropped.append(node)  # private constants would give answers away (unless still used)
                continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            _stub_body(node)
        elif isinstance(node, ast.ClassDef):
            kept: list[ast.stmt] = []
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    private = n.name.startswith("_") and not (
                        n.name.startswith("__") and n.name.endswith("__")
                    )
                    if private:
                        continue
                    _stub_body(n)
                kept.append(n)
            node.body = kept or [ast.Pass()]
        new_body.append(node)
    # re-insert dropped private constants that surviving code still references (to a fixpoint)
    changed = True
    while changed:
        used = {n.id for st in new_body for n in ast.walk(st) if isinstance(n, ast.Name)}
        changed = False
        for node in list(dropped):
            tg = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in used for t in tg):
                dropped.remove(node)
                new_body.append(node)
                changed = True
    order = {id(n): i for i, n in enumerate(originals)}
    new_body.sort(key=lambda n: order.get(id(n), 0))
    tree.body = new_body
    out = ast.unparse(ast.fix_missing_locations(tree))
    # ast.unparse puts the module docstring in quotes on one line; restore a triple-quoted block
    doc = ast.get_docstring(ast.parse(source), clean=False)
    if doc is not None:
        first = ast.unparse(ast.Expr(ast.Constant(doc)))
        safe = doc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        out = out.replace(first, '"""' + safe + '"""', 1)
    return _space_defs(out) + "\n"


def _space_defs(code: str) -> str:
    lines = code.split("\n")
    out: list[str] = []
    for ln in lines:
        top_level_def = ln.startswith(("def ", "async def ", "class ", "@"))
        if top_level_def and out and out[-1].strip() and not out[-1].startswith("@"):
            out.extend(["", ""])
        out.append(ln)
    return "\n".join(out)


def generate_stubs(module: Module, force: bool = False) -> list[Path]:
    written: list[Path] = []
    if not module.solution_dir.exists():
        return written
    for sol in sorted(module.solution_dir.glob("ex*.py")):
        target = module.exercise_dir / sol.name
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(make_stub(sol.read_text(encoding="utf-8")), encoding="utf-8")
        written.append(target)
    if written:
        _format([str(p) for p in written])
    return written


def _format(paths: list[str]) -> None:
    import shutil
    import subprocess
    import sys

    ruff = shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")
    if ruff:
        subprocess.run([ruff, "check", "-q", "--select", "I,F401", "--fix", *paths], check=False)
        subprocess.run([ruff, "format", "-q", *paths], check=False)


CAPSTONE_GIVEN = {"models.py", "interfaces.py", "__init__.py"}


def generate_capstone_stubs(force: bool = False) -> list[Path]:
    """Mirror solutions/capstone/src -> capstone/src; given files copied, the rest stubbed."""
    from .paths import ROOT, SOLUTIONS_DIR

    src, dst = SOLUTIONS_DIR / "capstone" / "src", ROOT / "capstone" / "src"
    written: list[Path] = []
    for sol in sorted(src.rglob("*.py")):
        target = dst / sol.relative_to(src)
        if target.exists() and not force:
            continue  # never overwrite real work
        target.parent.mkdir(parents=True, exist_ok=True)
        text = sol.read_text(encoding="utf-8")
        target.write_text(text if sol.name in CAPSTONE_GIVEN else make_stub(text), encoding="utf-8")
        written.append(target)
    if written:
        _format([str(p) for p in written if p.name not in CAPSTONE_GIVEN])
    return written
