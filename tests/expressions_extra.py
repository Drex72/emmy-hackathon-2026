"""Extra practice problems in the five categories: new shapes and ops, the same concepts.

Generate them (needs torch, only for tracing) with:

    python tools/generate.py tests/expressions_extra.py --out tests/problems

then check them with ./check_all.sh --dir tests/problems (the generated set is committed
there, so checking needs no torch).
"""

import math

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"
f64 = np.float64


def _erf(x):
    return np.vectorize(math.erf)(x)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _mm(a, b):
    return (a.astype(f64) @ b.astype(f64)).astype(np.float32)


def _attn(q, kt, v, scale):
    return (_softmax((q.astype(f64) @ kt.astype(f64)) / scale) @ v.astype(f64)).astype(np.float32)


OPEN = [
    # ---- pointwise -------------------------------------------------------
    dict(id="xp01_odd_add", code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="xp02_sigmoid", code="torch.sigmoid(torch.randn(4096))",
         ref=lambda x: [(1 / (1 + np.exp(-x.astype(f64)))).astype(np.float32)], rtol=1e-5, atol=1e-6),
    dict(id="xp03_gelu", code="torch.nn.functional.gelu(torch.randn(2048))",
         ref=lambda x: [(0.5 * x.astype(f64) * (1 + _erf(x.astype(f64) / np.sqrt(2)))).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="xp04_tanh_mul", code="x = torch.randn(3000); y = torch.randn(3000); torch.tanh(x) * y",
         ref=lambda x, y: [(np.tanh(x.astype(f64)) * y).astype(np.float32)], rtol=1e-5, atol=1e-6),
    dict(id="xp05_bcast_row", code="torch.randn(64, 128) + torch.randn(128)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="xp06_bcast_col", code="torch.randn(64, 1) * torch.randn(64, 96)",
         ref=lambda a, b: [a * b], rtol=0, atol=0),
    dict(id="xp07_where", code="x = torch.randn(2048); y = torch.randn(2048); torch.where(x > 0, x, y * 0.5)",
         ref=lambda x, y: [np.where(x > 0, x, y * np.float32(0.5))], rtol=1e-6, atol=1e-6),
    dict(id="xp08_silu_log", code="x = torch.randn(33, 77); torch.nn.functional.silu(x) + torch.log(torch.abs(x) + 1.0)",
         ref=lambda x: [((x.astype(f64) / (1 + np.exp(-x.astype(f64)))) + np.log(np.abs(x.astype(f64)) + 1)).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="xp09_clamp_pow", code="x = torch.randn(4096); torch.clamp(x, -0.5, 0.5) + x ** 2",
         ref=lambda x: [(np.clip(x.astype(f64), -0.5, 0.5) + x.astype(f64) ** 2).astype(np.float32)], rtol=1e-6, atol=1e-6),
    dict(id="xp10_exp_rsqrt", code="x = torch.randn(512, 8); torch.exp(x) * torch.rsqrt(torch.abs(x) + 1.0)",
         ref=lambda x: [(np.exp(x.astype(f64)) / np.sqrt(np.abs(x.astype(f64)) + 1)).astype(np.float32)], rtol=1e-5, atol=1e-6),

    # ---- reductions ------------------------------------------------------
    dict(id="xr01_max", code="torch.randn(4096).amax()",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="xr02_odd_sum", code="torch.randn(3000).sum()",
         ref=lambda x: [x.astype(f64).sum().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="xr03_row_mean", code="torch.randn(128, 1000).mean(-1)",
         ref=lambda x: [x.astype(f64).mean(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="xr04_row_min", code="x = torch.randn(32, 256); -((-x).amax(-1))",
         ref=lambda x: [x.min(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="xr05_col_sum", code="torch.randn(256, 64).sum(0)",
         ref=lambda x: [x.astype(f64).sum(0).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="xr06_sumsq", code="x = torch.randn(8192); (x * x).sum()",
         ref=lambda x: [(x.astype(f64) ** 2).sum().astype(np.float32)], rtol=1e-4, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="xr07_unpinned_sum", code="torch.randn(1 << 20).sum()",
         ref=lambda x: [x.astype(f64).sum().astype(np.float32)], rtol=1e-3, atol=1e-2),
    dict(id="xr08_unpinned_rows", code="torch.randn(512, 2048).sum(-1)",
         ref=lambda x: [x.astype(f64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-3),

    # ---- matmul ----------------------------------------------------------
    dict(id="xm01_square_256", code="torch.randn(256, 256) @ torch.randn(256, 256)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),
    dict(id="xm02_tall", code="torch.randn(1024, 64) @ torch.randn(64, 32)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),
    dict(id="xm03_odd", code="torch.randn(100, 70) @ torch.randn(70, 50)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),
    dict(id="xm04_splitk", code="torch.randn(32, 1024) @ torch.randn(1024, 32)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),
    dict(id="xm05_wide", code="torch.randn(64, 128) @ torch.randn(128, 512)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),

    dict(id="xm06_bias_relu", code="a = torch.randn(128, 64); b = torch.randn(64, 96); c = torch.randn(96); torch.relu(a @ b + c)",
         ref=lambda a, b, c: [np.maximum(_mm(a, b) + c, 0)], rtol=1e-3, atol=1e-3),
    dict(id="xm07_batched", code="torch.randn(4, 64, 32) @ torch.randn(4, 32, 48)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),
    dict(id="xm08_big_k", code="torch.randn(64, 2048) @ torch.randn(2048, 64)",
         ref=lambda a, b: [_mm(a, b)], rtol=1e-3, atol=1e-3),

    # ---- fused -----------------------------------------------------------
    dict(id="xf07_layernorm_fn", code="torch.nn.functional.layer_norm(torch.randn(32, 256), (256,))",
         ref=lambda x: [((x.astype(f64) - x.astype(f64).mean(-1, keepdims=True))
                         / np.sqrt(x.astype(f64).var(-1, keepdims=True) + 1e-5)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="xf08_sum_sigmoid", code="x = torch.randn(48, 200); x * torch.sigmoid(x).sum(-1, keepdim=True)",
         ref=lambda x: [(x * (1 / (1 + np.exp(-x.astype(f64)))).sum(-1, keepdims=True)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="xf01_layernorm",
         code="x = torch.randn(64, 512); (x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True) + 1e-5)",
         ref=lambda x: [((x.astype(f64) - x.astype(f64).mean(-1, keepdims=True))
                         / np.sqrt(x.astype(f64).var(-1, keepdims=True) + 1e-5)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="xf02_softmax_wide", code="torch.softmax(torch.randn(32, 1000), dim=-1)",
         ref=lambda x: [_softmax(x.astype(f64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="xf03_rmsnorm_768",
         code="x = torch.randn(128, 768); w = torch.randn(768); x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w",
         ref=lambda x, w: [(x / np.sqrt((x.astype(f64) ** 2).mean(-1, keepdims=True) + 1e-6) * w).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="xf04_log_softmax",
         code="x = torch.randn(64, 512); m = x.amax(-1, keepdim=True); x - m - torch.log(torch.exp(x - m).sum(-1, keepdim=True))",
         ref=lambda x, *_: [np.log(_softmax(x.astype(f64))).astype(np.float32)], rtol=1e-3, atol=1e-4),
    dict(id="xf05_center_max", code="x = torch.randn(64, 300); x - x.amax(-1, keepdim=True)",
         ref=lambda x: [x - x.max(-1, keepdims=True)], rtol=1e-6, atol=1e-6),
    dict(id="xf06_softmax_cols", code="torch.softmax(torch.randn(128, 64), dim=0)",
         ref=lambda x: [_softmax(x.astype(f64), axis=0).astype(np.float32)], rtol=1e-3, atol=1e-5),

    # ---- attention-like --------------------------------------------------
    dict(id="xa01_attn_small",
         code="q = torch.randn(32, 16); kt = torch.randn(16, 64); v = torch.randn(64, 16); torch.softmax(q @ kt / 4.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn(q, kt, v, 4.0)], rtol=1e-3, atol=1e-4),
    dict(id="xa02_attn_256",
         code="q = torch.randn(256, 64); kt = torch.randn(64, 256); v = torch.randn(256, 64); torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
    dict(id="xa03_attn_rect",
         code="q = torch.randn(48, 32); kt = torch.randn(32, 80); v = torch.randn(80, 24); torch.softmax(q @ kt * 0.125, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn(q, kt, v, 8.0)], rtol=1e-3, atol=1e-4),
    dict(id="xa04_attn_batched",
         code="q = torch.randn(2, 64, 32); kt = torch.randn(2, 32, 64); v = torch.randn(2, 64, 32); torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [_attn(q, kt, v, 5.656854)], rtol=1e-3, atol=1e-4),
    dict(id="xa05_attn_causalish",
         code="q = torch.randn(64, 32); kt = torch.randn(32, 64); v = torch.randn(64, 32); torch.softmax(torch.tanh(q @ kt * 0.1), dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax(np.tanh((q.astype(f64) @ kt.astype(f64)) * 0.1)) @ v.astype(f64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
]
