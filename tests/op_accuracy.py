#!/usr/bin/env python3
"""Measure the accuracy of every unary elementwise op the compiler supports, on the GPU.

Takes the p03 pointwise IR (``mul = y * relu(x)``), swaps ``relu`` for the op
under test, compiles it with kernelc, runs it on the hosted runner with chosen
inputs (y = 1) and reports the worst error against a float64 reference rounded
to float32, in ulps and as a relative error.

    python tests/op_accuracy.py            # all ops
    python tests/op_accuracy.py exp tanh   # some
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernelc.codegen import compile_kernel  # noqa: E402
from kernelc.loader import load  # noqa: E402

RUNNER = "http://runner.riftstack.ai:48701"
PROBLEM = ROOT / "problems" / "p03_relu_mul"
N = 2048


def _erf(x):
    return np.vectorize(math.erf)(x)


REFERENCE = {
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "reciprocal": lambda x: 1 / x,
    "tanh": np.tanh,
    "sin": np.sin,
    "cos": np.cos,
    "erf": _erf,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "silu": lambda x: x / (1 + np.exp(-x)),
    "softplus": lambda x: np.logaddexp(0, x),
    "gelu": lambda x: 0.5 * x * (1 + _erf(x / np.sqrt(2))),
    "gelu_tanh": lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    "abs": np.abs,
    "negative": np.negative,
    "square": np.square,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
}

# inputs per op: a dense sweep over the interesting range plus random normals
POSITIVE = {"log", "log2", "log10", "sqrt", "rsqrt"}


def inputs(op: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if op in POSITIVE:
        x = np.concatenate([np.geomspace(1e-30, 1e30, N // 2), np.abs(rng.standard_normal(N // 2)) + 1e-3])
    elif op == "log1p":
        x = np.concatenate([np.linspace(-0.99, 3, N // 2), rng.standard_normal(N // 2) * 1e-3])
    elif op in ("sin", "cos"):
        x = np.concatenate([np.linspace(-50, 50, N // 2), rng.standard_normal(N // 2)])
    else:
        x = np.concatenate([np.linspace(-12, 12, N // 2), rng.standard_normal(N // 2) * 3])
    return x.astype(np.float32)


def ulp_error(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref32 = ref.astype(np.float32)
    spacing = np.spacing(np.abs(ref32)).astype(np.float64)
    err = np.abs(got.astype(np.float64) - ref)
    both_nan = np.isnan(got) & np.isnan(ref)
    same_inf = np.isinf(got) & np.isinf(ref) & (np.sign(got) == np.sign(ref))
    out = np.where(both_nan | same_inf, 0.0, err / np.maximum(spacing, np.finfo(np.float32).tiny))
    return np.nan_to_num(out, nan=np.inf)


def run(op: str) -> tuple[float, float, float]:
    ir = json.loads(next(PROBLEM.glob("*.ir.json")).read_text())
    body = ir["nodes"]["mul"]["op_fields"]["body"]
    ir["nodes"]["mul"]["op_fields"]["body"] = [b.replace("ElementwiseImpl('relu')", f"ElementwiseImpl('{op}')") for b in body]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k_mul_pointwise.ir.json"
        path.write_text(json.dumps(ir))
        (Path(tmp) / "meta.json").write_text((PROBLEM / "meta.json").read_text())
        ptx = compile_kernel(load(str(path)))
    meta = json.loads((PROBLEM / "meta.json").read_text())["kernels"][0]
    x = inputs(op)
    args = [
        {"name": "x", "dtype": "f32", "shape": [N], "data": x.tolist()},
        {"name": "y", "dtype": "f32", "shape": [N], "data": [1.0] * N},
        {"name": "mul", "dtype": "f32", "shape": [N], "output": True},
    ]
    r = requests.post(f"{RUNNER}/run", json={"ptx": ptx, "entry": meta["entry"], "grid": meta["grid"], "block": meta["block"],
                                             "smem_bytes": 0, "zero_outputs": [], "args": args, "max_return": N}, timeout=120).json()
    if not r.get("ok"):
        raise RuntimeError(r.get("stderr", r))
    got = np.array(r["outputs"]["mul"]["values"], dtype=np.float32)
    with np.errstate(all="ignore"):
        ref = REFERENCE[op](x.astype(np.float64))
    ulps = ulp_error(got, ref)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    finite = np.isfinite(ref) & (np.abs(ref) > 1e-30)
    absolute = np.abs(got - ref)
    return float(np.max(ulps)), float(np.max(rel[finite])) if finite.any() else 0.0, float(np.nanmax(np.where(np.isfinite(ref), absolute, 0)))


def main() -> int:
    ops = sys.argv[1:] or list(REFERENCE)
    print(f"{'op':<12} {'max ulp':>10} {'max rel':>12} {'max abs':>12}")
    worst = 0.0
    for op in ops:
        ulp, rel, ab = run(op)
        worst = max(worst, rel)
        print(f"{op:<12} {ulp:>10.2f} {rel:>12.3e} {ab:>12.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
