#!/usr/bin/env python3
"""Static contract checks for the flood configs and the vendored mmseg package.

Every check here exists because the corresponding bug actually shipped and made
a clean checkout unrunnable. None of them need torch or mmcv-full, so they run
in seconds on CI, where installing this codebase's 2020-era stack would not.

    python tools/check_configs.py

Checks
------
1. Package exports resolve. Every name `mmseg/datasets/pipelines/__init__.py`
   imports from a sibling module is actually defined there, and everything in
   __all__ is imported. Deleting a class that __init__ still imports breaks
   `from mmseg.apis import init_segmentor` with an ImportError far from the
   edit.

2. No `style=` in a backbone. It is a ResNet-ism; the vendored mit_* backbones
   reject it:
       TypeError: mit_b0: __init__() got an unexpected keyword argument 'style'

3. `in_chans` is declared and matches the loader. It defaults to 3, so a
   five-band config that omits it meets a three-channel stem:
       RuntimeError: weight of size [32, 3, 7, 7], expected input[1, 5, ...]
       to have 3 channels, but got 5 channels instead

4. Normalisation is declared, and `meanstd` has one statistic per band. A
   three-entry ImageNet mean against a five-band model normalises RGB and
   leaves thermal and NIR at raw 0-255, silently.

5. Everything compiles.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINES = ROOT / "mmseg" / "datasets" / "pipelines"

#: loader in a config's pipeline -> channels its backbone must declare
LOADER_CHANNELS = {
    "Load_5band_ImageFromFile": 5,
    "Load_LWIR_ImageFromFile": 1,
}
DEFAULT_CHANNELS = 3

failures: list[str] = []


def fail(where: str, msg: str) -> None:
    failures.append(f"{where}: {msg}")


# --------------------------------------------------------------------------- 1
def check_exports() -> None:
    init = PIPELINES / "__init__.py"
    tree = ast.parse(init.read_text())
    imported: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        sibling = PIPELINES / f"{node.module}.py"
        names = {a.name for a in node.names}
        imported |= names
        if not sibling.exists():
            fail(str(init.relative_to(ROOT)), f"imports from missing module {node.module!r}")
            continue
        defined = {
            n.name for n in ast.walk(ast.parse(sibling.read_text()))
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        } | {
            t.id for n in ast.walk(ast.parse(sibling.read_text()))
            if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
        }
        for missing in sorted(names - defined):
            fail(f"mmseg/datasets/pipelines/{node.module}.py",
                 f"__init__.py imports {missing!r} but it is not defined here")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            exported = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
            for missing in sorted(exported - imported):
                fail(str(init.relative_to(ROOT)), f"__all__ lists {missing!r} but it is never imported")


# ------------------------------------------------------------------- 2, 3, 4
def load_config(path: Path) -> dict:
    """Resolve a config and its _base_ chain. Child keys win, dicts merge."""
    ns: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102 - trusted repo files
    ns.pop("__builtins__", None)

    bases = ns.pop("_base_", []) or []
    if isinstance(bases, str):
        bases = [bases]

    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, load_config((path.parent / b).resolve()))
    return deep_merge(merged, ns)


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_dicts(v)


def check_config(path: Path) -> None:
    where = str(path.relative_to(ROOT))
    try:
        cfg = load_config(path)
    except Exception as e:                                  # noqa: BLE001
        fail(where, f"could not be loaded: {type(e).__name__}: {e}")
        return

    backbone = (cfg.get("model") or {}).get("backbone") or {}
    btype = str(backbone.get("type", ""))

    if "style" in backbone:
        fail(where, f"backbone {btype!r} passes style={backbone['style']!r}; "
                    f"the vendored mit_* backbones reject it")

    loaders = {d["type"] for d in walk_dicts(cfg) if d.get("type") in LOADER_CHANNELS}
    expected = LOADER_CHANNELS[next(iter(loaders))] if len(loaders) == 1 else DEFAULT_CHANNELS
    if len(loaders) > 1:
        fail(where, f"pipelines mix loaders {sorted(loaders)}; cannot infer band count")
        return

    if btype.startswith("mit_"):
        if "in_chans" not in backbone:
            fail(where, f"backbone does not declare in_chans; it defaults to "
                        f"{DEFAULT_CHANNELS} and this config loads {expected}-band input")
        elif backbone["in_chans"] != expected:
            fail(where, f"in_chans={backbone['in_chans']} but the pipeline loads "
                        f"{expected}-band input ({sorted(loaders) or 'default RGB'})")

    for d in walk_dicts(cfg):
        if d.get("type") != "Normalize_5band":
            continue
        if "method" not in d:
            fail(where, "Normalize_5band does not declare method=; it must match what the "
                        "checkpoint was trained with")
        elif d["method"] == "meanstd":
            n = backbone.get("in_chans", expected)
            for k in ("mean", "std"):
                if len(d.get(k, [])) != n:
                    fail(where, f"Normalize_5band method='meanstd' has {len(d.get(k, []))} "
                                f"{k} values for {n} bands; one per band is required")


# --------------------------------------------------------------------------- 5
def check_compiles() -> None:
    for p in sorted(ROOT.glob("mmseg/**/*.py")) + sorted(ROOT.glob("tools/*.py")) + \
             sorted(ROOT.glob("local_configs/**/*.py")) + sorted(ROOT.glob("*.py")):
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            fail(str(p.relative_to(ROOT)), f"does not parse: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="local_configs/segformer/*/*flood*.py",
                    help="glob of configs to check")
    a = ap.parse_args()

    check_exports()
    check_compiles()

    configs = sorted(ROOT.glob(a.configs))
    if not configs:
        fail("config glob", f"matched nothing: {a.configs!r}")
    for c in configs:
        check_config(c)

    print(f"checked {len(configs)} configs, the mmseg package exports, and syntax")
    if failures:
        print(f"\n{len(failures)} problem(s):\n")
        for f in failures:
            print(f"  {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
