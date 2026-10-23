"""Load a Kernel IR json and the launch contract that ships next to it.

The kernel body is deserialized through Emmy's own classes (``Graph.from_dict``);
everything else -- buffer shapes, dtypes, constant values -- is read from the
plain graph level of the json. The parameter order comes from ``meta.json``'s
``arg_order`` when the file is present (it is the contract), and falls back to
the KernelOp's inputs followed by its outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from emmy.compiler.graph import Graph


@dataclass
class Param:
    name: str
    dtype: str  # f32 | f16 | f64 | i32 | i64
    scalar: bool = False  # passed by value instead of as a pointer


@dataclass
class KernelSource:
    name: str
    body: tuple
    params: list[Param]
    shapes: dict[str, tuple[int, ...]]  # graph buffer -> shape
    dtypes: dict[str, str]  # graph buffer -> dtype name
    constants: dict[str, float] = field(default_factory=dict)  # ConstantOp buffer -> value
    knobs: dict = field(default_factory=dict)


def _find_kernel(graph: Graph):
    kernels = [(n, node.op) for n, node in graph.nodes.items() if type(node.op).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise ValueError(f"expected exactly one KernelOp in the IR, found {len(kernels)}")
    return kernels[0]


def _meta_kernel(ir_path: Path, entry: str) -> dict | None:
    meta_path = ir_path.parent / "meta.json"
    if not meta_path.exists():
        return None
    kernels = json.loads(meta_path.read_text()).get("kernels", [])
    for k in kernels:
        if k.get("entry") == entry:
            return k
    return kernels[0] if len(kernels) == 1 else None


def load(path: str) -> KernelSource:
    ir_path = Path(path)
    raw = json.loads(ir_path.read_text())
    graph = Graph.from_dict(raw)
    _, op = _find_kernel(graph)

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    constants: dict[str, float] = {}
    for node_name, node in raw["nodes"].items():
        for out in node.get("outputs", []):
            shapes[out["name"]] = tuple(int(d) for d in out.get("shape", []))
            dtypes[out["name"]] = out.get("dtype", "f32")
        if node.get("op") == "ConstantOp":
            value = node.get("op_fields", {}).get("value")
            if isinstance(value, (int, float)):
                constants[node_name] = float(value)

    name = op.name
    meta = _meta_kernel(ir_path, name)
    if meta is not None:
        by_name = {a["name"]: a for a in meta.get("args", [])}
        params = []
        for arg in meta["arg_order"]:
            spec = by_name.get(arg, {})
            params.append(Param(arg, spec.get("dtype", dtypes.get(arg, "f32")), bool(spec.get("scalar", False))))
            if "shape" in spec and not spec.get("scalar"):
                shapes[arg] = tuple(int(d) for d in spec["shape"])
    else:
        params = [Param(n, dtypes.get(n, "f32")) for n in (*op.inputs, *op.outputs)]

    return KernelSource(
        name=name,
        body=tuple(op.body),
        params=params,
        shapes=shapes,
        dtypes=dtypes,
        constants=constants,
        knobs=dict(getattr(op, "knobs", {}) or {}),
    )
