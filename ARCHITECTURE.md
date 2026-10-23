# Kernel IR — formal specification

Kernel IR is the last dialect before CUDA source. One `KernelOp` is one `__global__` function: a body of statements
in which every scheduling decision has already been made. Thread binding, shared memory, barriers, asynchronous
copies, cross-thread combines and tensor-core fragments are all explicit statements. What is left is a program that
renders to CUDA one statement at a time.

```
TileOp ──lowering/kernel (010_materialize + peepholes)──▶ KernelOp ──lowering/cuda (render_kernelop)──▶ CudaOp
```

This file is the specification of the dialect: its abstract syntax (§2), its name spaces and scoping (§3), its types
(§4), the well-formedness rules a `KernelOp` must satisfy (§5), its execution model (§6), the kernel signature and
launch it implies (§7), and how each statement renders (§8). §9 says which rules are checked by code and which are
contracts the lowering keeps. The node definitions are in `ir.py`, the renderer is `render.py`, and the shared
statements and expressions come from `ir/stmt/` and `ir/expr.py`. How `TileOp` becomes a `KernelOp` is the
[kernel lowering ARCHITECTURE.md](../../pipeline/passes/lowering/kernel/ARCHITECTURE.md).

## 1. Notation

- `x ::= A | B` is a grammar production. `X*` is a sequence of zero or more, `X?` is optional, `(X, Y)` is a tuple.
- `ℕ` is a natural number, `ℤ` an integer, `str` an identifier. `Expr` is §2.5. `DType` is §4.
- Field names and defaults are the dataclass fields in code. A field written `= v` has default `v`.
- `Γ` is a typing scope: a finite map from names to the kind of thing they name (§3).
- `Γ ⊢ b ⊣ Γ'` reads "under scope `Γ`, body `b` is well formed and leaves scope `Γ'` to its successors".
- `⟦s⟧` is the meaning of statement `s` as a transition on machine state (§6).

## 2. Abstract syntax

### 2.1 Program

```
KernelOp ::= KernelOp(name: str,
                      body: Body,
                      zero_delegated: (str*) = (),
                      serial: (Axis*) = (),
                      inputs: {str ↦ Tensor}, outputs: {str ↦ Tensor},   -- inherited from Op, derived (§7)
                      knobs: {str ↦ value})                                -- recorded peephole decisions only

Body     ::= Stmt*                                  -- an immutable tuple; `Body.iter()` is pre-order, deep
Axis     ::= Axis(name: str, extent: Dim, window: Window?)
Dim      ::= ℕ | symbolic size (e.g. Dim("seq_len"), a product with such a factor)
```

A `KernelOp` is a frozen graph op (`BodyOp`). `Axis.window` records which parent axis an axis was carved from; it
does not take part in equality and has no runtime meaning in Kernel IR.

### 2.2 Statements

Statements fall into seven groups. Each one is an immutable, hashable dataclass.

```
Stmt ::= Control | Scalar | Memory | Barrier | Async | Combine | Fragment | Wgmma

-- Control: statements that own nested bodies ----------------------------------------------------------------------
Control ::= Tile(axes: (Axis*), body: Body,
                 block_threads: ℕ? = None, aux_threads: ℕ = 0,
                 raster_axes: (str, str)? = None, raster_group: ℕ? = None, raster_orient: "m"|"n" = "m",
                 spread: bool = False)
          | Loop(axis: Axis, body: Body, unroll: bool = False, seed: bool = True)
          | StridedLoop(axis: Axis, start: Expr, step: Expr, body: Body,
                        unroll: bool = False, end: Expr? = None, seed: bool = True)
          | Cond(cond: Expr, body: Body, else_body: Body = ())

-- Scalar: pure SSA bindings (shared with Loop IR) -------------------------------------------------------------------
Scalar  ::= Assign(name: str, op: ElementwiseImpl, args: (str*), dtype: DType? = None)
          | Let(name: str, value: Expr, dtype: DType? = None)
          | Select(name: str, branches: (SelectBranch+))         SelectBranch ::= (value: str, select: Expr)
          | Init(name: str, identity: float, dtype: DType)
          | Accum(name: str, value: str, op: ElementwiseImpl = add, dtype: DType? = None,
                  axes: (str*) = (), base: str? = None)

-- Memory: reads and writes of global or shared buffers (shared with Loop IR) ----------------------------------------
Memory  ::= Load(names: (str+), input: str, index: (Expr*), dtype: DType?, carried: "" | "load" | "unpack" = "")
          | Write(output: str, index: (Expr*), values: (str+), value_dtype: DType?, atomic: bool, swizzle: Swz)
          | Carry(name: str, value: str, index: (Expr*), seed: float | str, dtype: DType? = None)
          | Pre(name: str, carrier: str, index: (Expr*))
          | ZeroPrologue(dst: str, words: ℕ)
          | Smem(name: str, extents: (ℕ*), dtype: CType = "float", align: ℕ = 0)

-- Barrier ------------------------------------------------------------------------------------------------------------
Barrier ::= Sync(barrier_id: ℕ = 0, count: ℕ? = None, warp: bool = False)

-- Async: transports from global into shared memory ------------------------------------------------------------------
Async   ::= CpAsyncCopy(smem: str, smem_index: (Expr*), src: str, src_index: (Expr*), nbytes: 4|8|16 = 4,
                        swizzle: Swz = "NONE", lane_index: (Expr*)? = None, lane_rows: ℕ = 0, valid: Expr? = None)
          | CpAsyncCommit()
          | CpAsyncWait(group: ℕ = 0)
          | TmaDescriptor(name: str, src_buf: str, src_shape: (ℕ*), box_extents: (ℕ*), swizzle: Swz = "NONE",
                          dtype: CType = "float")
          | TmaLoad(smem: str, smem_index: (Expr*), desc: str, coords: (Expr*), mbar: str, mbar_slot: Expr? = None)
          | MbarrierInit(mbar: str, count: ℕ = 1, slot: Expr? = None)
          | MbarrierArriveExpectTx(mbar: str, bytes_: ℕ, slot: Expr? = None)
          | MbarrierArrive(mbar: str, slot: Expr? = None)
          | MbarrierWait(mbar: str, phase: Expr, slot: Expr? = None)
          | SetMaxNReg(count: ℕ, direction: "inc" | "dec")

-- Combine: cross-thread reductions and scans ------------------------------------------------------------------------
Combine ::= WarpShuffle(state: (str+), state_b: (str+), combine_states: (Assign*), length: ℕ,
                        dtype: DType = f32, prefix: bool = False)
          | WarpBroadcast(state: (str+), length: ℕ)
          | TreeHalve(bufs: (str+), state: (str+), state_b: (str+), combine_states: (Assign*), length: ℕ,
                      tid_var: str, dtype: DType = f32, barrier_id: ℕ = 0, barrier_count: ℕ? = None,
                      inner: (str, ℕ)? = None)

-- Fragment: mma.sync register fragments ----------------------------------------------------------------------------
Fragment ::= RegFragment(name: str, role: "a"|"b"|"c"|…, shape: (ℕ, ℕ, ℕ), dtype: DType, count: ℕ = 1,
                         nregs: ℕ? = None)
           | LdmatrixLoad(frag: str, src_buffer: str, src_index: (Expr*), role: str, ldm: ℕ = 0, swizzle: Swz = "NONE",
                          staged: bool = True, gmem_guard: (Expr, Expr)? = None, k_zero: (Expr, Expr)? = None,
                          b_trans: bool = False, pair_frag: str? = None, byte_slab: bool = False,
                          scale_buffer: str? = None, scale_index: (Expr*) = (), scale_ldm: ℕ = 0,
                          fragment_layout: Atom = "m16n8k16", fragment_index: ℕ = 0)
           | BlockScaleLoad(frag: str, src_buffer: str, src_index: (Expr*), role: str, ldm: ℕ = 0)
           | MmaSyncPtx(c_frag: str, a_frag: str, b_frag: str, shape: (ℕ, ℕ, ℕ), ab_dtype: AB = "f16",
                        c_dtype: "f32"|"f16" = "f32", b_row_major: bool = False, sfa_frag: str? = None, sfb_frag: str? = None)
           | FragmentPromote(dst: str, src: str, fragment_layout: Atom = "m16n8k16")
           | FragmentRepack(frag: str, srcs: (str+), ab_dtype: AB = "f16", fragment_layout: Atom = "m16n8k16",
                            part: ℕ = 0, role: "a"|"b" = "a")
           | FragmentApply(out: str, op: ElementwiseImpl, args: (Operand*), kinds: (Kind*), in_place: bool = False,
                           layout: FragLayout = M16N8, post: (ElementwiseImpl*) = (), row_base: Expr? = None,
                           col_base: Expr? = None)
           | FragmentRowReduce(top: str, bot: str, frags: (str+), op: ElementwiseImpl, layout: FragLayout = M16N8,
                               dtype: DType = f32)
           | RegStore(dst_buffer: str, dst_index: (Expr*), frag: str, shape: (ℕ, ℕ, ℕ), ldm: ℕ = 0, ldn: ℕ = 0,
                      epilogue: Lambda? = None, extra_frags: (str*) = (), m_guard: (Expr, Expr)? = None,
                      n_guard: (Expr, Expr)? = None, atomic: bool = False, fragment_layout: Atom = "m16n8k16",
                      volta_interleaved: bool = False, fragment_index: (ℕ, ℕ) = (0, 0), swizzle: Swz = "NONE",
                      row_dim: ℕ? = None, col_dim: ℕ? = None, run: (RegStore*) = ())

-- Wgmma: Hopper warp-group tensor-core cells ------------------------------------------------------------------------
Wgmma   ::= WgmmaDescriptor(name: str, smem: str, smem_index: Expr?, swizzle: Swz, lbo_bytes: ℕ, sbo_bytes: ℕ)
          | WgmmaMma(c_frags: (str+), b_desc: str, shape: (ℕ, ℕ, ℕ), ab_dtype: AB = "f16", a_desc: str? = None,
                     a_frag: str? = None, scale_d: 0|1 = 1, trans_a: 0|1 = 0, trans_b: 0|1 = 0)
          | WgmmaFence() | WgmmaCommit() | WgmmaWait(group: ℕ = 0)
```

### 2.3 Auxiliary domains

```
Swz      ::= "NONE" | "B32" | "B64" | "B128" | "B32@k" | "B64@k" | "B128@k" | "V70A" | "V70B"     (k ∈ ℕ)
Atom     ::= "m16n8k16" | "m8n8k4" | "wgmma"                         -- C-fragment layout names (frag_layout)
AB       ::= "f16" | "bf16" | "e4m3" | "e5m2" | e2m1 (block-scaled)  -- the PTX multiplicand type field
Kind     ::= FRAG | ROW | UNIFORM | COORD | GMEM
Operand  ::= str                     -- FRAG, or UNIFORM by name
           | (str, str)              -- ROW: the two per-row SSA names (rows g and g+8)
           | Expr                    -- COORD predicate, or UNIFORM literal
           | (str, (Expr*))          -- GMEM: (buffer, index) load template
FragLayout ::= (n_elems: ℕ, elem_row: (ℕ*), reduce_xors: (ℕ*), lane_decl: str, lane_names: (str*),
                row_off: (Expr*), col_off: (Expr*))     -- exactly two instances: M16N8 and M8N8K4
Lambda   ::= a pure function whose body is Load / Assign / Let / Select statements (ir/pure/lam.py)
```

`Swz` names an address permutation of a shared slab. `B32`/`B64`/`B128` XOR the 16-byte chunk index with the row
index under the masks `0x1`/`0x3`/`0x7` (`LDMATRIX_SWIZZLE_XOR`). `@k` overrides the element shift with `k` for a
slab whose row is wider than one swizzle atom. `V70A` / `V70B` are the Volta crosswise and B-congruous layouts.

### 2.4 Elementwise operations

`ElementwiseImpl` is a named scalar operation (`add`, `multiply`, `maximum`, `exp`, `where`, `copy`, casts,
`from_f8e4m3`, …). Its name fixes its arity, its numpy reference meaning, and its algebraic traits: `commutative`,
`associative` and `identity` (`add ↦ 0`, `multiply ↦ 1`, `maximum ↦ −1e30`, `minimum ↦ 1e30`). Unknown names are
rejected at construction.

### 2.5 Expressions

Expressions are side-effect-free index and predicate terms.

```
Expr ::= Var(name: str)
       | Literal(value: ℤ | float | bool, dtype: "int" | "float" | … = "float")
       | BinaryExpr(op: BinOp, left: Expr, right: Expr)
       | TernaryExpr(cond: Expr, if_true: Expr, if_false: Expr)
       | FuncCallExpr(name: ElementwiseImpl name, args: (Expr*))
       | CastExpr(dtype: CType, expr: Expr)
       | Builtin(name: "threadIdx.x" | "blockIdx.x" | "blockDim.x" | …)
       | FlatIndex(buffer: str, index: (Expr*))
BinOp ::= + | - | * | / | // | % | < | <= | > | >= | == | != | && | || | & | | | ^ | << | >>
```

`/` and `//` on integers are floor division. `FlatIndex(b, i)` is the row-major element offset of coordinate `i` in
buffer `b`, computed from `b`'s shape when the kernel is rendered; it appears only as the value of a `Let`.

### 2.6 Statements that do not occur

These Loop IR statements are not Kernel IR. A `KernelOp` built by `lowering/kernel` never holds them:

- `Affine` — `010_materialize` spells each one as binary `Assign`s over `Let`-bound literals. Kernel IR evaluates
  one operation at a time.
- `Call` — subroutine calls expand before a `LoopOp` is validated.

## 3. Names and scoping

### 3.1 Name spaces

A Kernel IR program uses six kinds of name. They share one identifier space in the rendered C function, so a name
must not be reused across kinds in one C scope.

| Kind | Bound by | Scope |
|---|---|---|
| global buffer `g` | the graph (`KernelOp.inputs` / `outputs`); also `ZeroPrologue.dst` | the whole kernel |
| local buffer `s` | `Smem` (an mbarrier is an `Smem` of C type `unsigned long long`) | the rest of the enclosing body |
| descriptor `d` | `TmaDescriptor.name` (a kernel parameter); `WgmmaDescriptor.name` (a local) | kernel / rest of body |
| axis variable `a` | `Tile.axes`, `Loop.axis`, `StridedLoop.axis`, `KernelOp.serial` | the owning body / whole kernel |
| SSA value `v` | the `defines()` of a statement (below) | the rest of the enclosing body |
| symbolic size `n` | `Dim` names in extents and buffer shapes, passed as `int` parameters | the whole kernel |

The statement that defines each SSA value:

| Statement | Defines |
|---|---|
| `Load` | `names` (except `carried = "load"`, which re-assigns the packed vector of names an earlier `Load` declared) |
| `Assign`, `Let`, `Select`, `Init`, `Pre` | `name` |
| `Accum` | `name`, visible after the enclosing reduce loop closes |
| `Carry` | `name` (the carried state) |
| `RegFragment` | `name` (a fragment family when `count > 1`) |
| `FragmentApply` | `out`, unless `in_place` |
| `FragmentRowReduce` | `top`, `bot` |
| `FragmentPromote`, `FragmentRepack`, `LdmatrixLoad`, `BlockScaleLoad`, `MmaSyncPtx`, `WgmmaMma` | they assign fragments that a `RegFragment` declared |
| `WgmmaDescriptor` | `name` |
| `WarpShuffle`, `WarpBroadcast`, `TreeHalve` | nothing new; they re-assign `state` in place |

Reserved names, supplied by the renderer and never bound in the body:

- `lane = threadIdx.x & 31`, `warp = threadIdx.x >> 5`, declared at function entry when the body reads them.
- `__frow`, `__fcol` — a fragment element's absolute row and column inside a `FragmentApply` COORD or GMEM template.
- `__M__`, `__N__` — a fragment element's row and column offset inside a `RegStore` epilogue.
- `_gid` and the raster temporaries `_rsub`, `_rgrp`, `_rsz` — declared by `Tile`.
- The `lane_names` of a `FragLayout` (`_g`, `_t` for M16N8; `_vl`, `_vq`, `_vr`, `_vc` for M8N8K4).
- `_r<depth>` — the induction variables `100_loopify` introduces.

### 3.2 Scoping rules

Let `Γ` hold every name in scope. A body is checked left to right:

```
Γ ⊢ () ⊣ Γ

Γ ⊢ s ⊣ Γ₁     Γ₁ ⊢ b ⊣ Γ₂
──────────────────────────       (sequencing)
Γ ⊢ (s, b) ⊣ Γ₂
```

A leaf statement `s` is well scoped when everything it reads is in scope, and it extends the scope with what it
defines:

```
free(s) ⊆ dom(Γ)        defines(s) ∩ declared_here(Γ) = ∅
────────────────────────────────────────────────────────       (leaf)
Γ ⊢ s ⊣ Γ ∪ defines(s)
```

`free(s)` is the SSA names in `s.deps()` plus every free variable of `s.exprs()` (`free_names`). `declared_here(Γ)`
is the names declared in the current C scope. An inner body may shadow an outer name. Re-declaring a name in the same
scope is an error, except one case: a repeat that binds the same name to an identical value is dropped by
`010_materialize` before any other pass sees it.

Statements with nested bodies open a new scope. Only `Accum` targets cross back out:

```
Γ ∪ {a} ⊢ body ⊣ Γ'        A = { x.name | x ∈ Accum statements directly folded by this loop }
──────────────────────────────────────────────────────────────────       (Loop, StridedLoop)
Γ ⊢ Loop(a, body) ⊣ Γ ∪ A

Γ ⊢ body ⊣ Γ₁        Γ ⊢ else_body ⊣ Γ₂        A = Accum targets of either body
──────────────────────────────────────────────────────────────────       (Cond)
Γ ⊢ Cond(c, body, else_body) ⊣ Γ ∪ A           with free(c) ⊆ dom(Γ)

Γ ∪ {a₁ … aₙ} ⊢ body ⊣ Γ'
──────────────────────────────                                            (Tile)
Γ ⊢ Tile((a₁ … aₙ), body) ⊣ Γ
```

`Smem` declarations are visible to the rest of their body and to everything nested in it. Each `Smem.name` is unique
in the kernel; `KernelOp.smem_buffers` keys them by name.

### 3.3 Closed program

The kernel's body must read nothing its launch does not supply:

```
free(KernelOp.body) ⊆ inputs ∪ outputs ∪ serial ∪ {extent names} ∪ {reserved names}
```

`010_materialize` asserts this (`_unbound_names`). A name outside that set is a compiler bug, and the assertion names
the kernel and the value.

## 4. Types

### 4.1 Value types

`DType` is the compiler's element type (`compiler/dtype.py`): `f32`, `f16`, `bf16`, `f8e4m3`, `f8e5m2`, `f4e2m1`,
the integer types and `bool`. `CType` is a C spelling of one (`"float"`, `"__half"`, `"unsigned long long"`, …), used
where a statement declares storage (`Smem`, `TmaDescriptor`).

Every SSA value has one `DType`. `030_stamp_types` resolves it once and writes it on the statement:

- `Load.dtype` is the element type of its buffer.
- `Assign.dtype` is set when it differs from `promote(args)`; otherwise the promoted type applies.
- `Select` takes the common type of its branches.
- `Accum` keeps its input type for a selecting fold (`maximum`, `minimum`) and promotes to `f32` for an accumulating
  fold (`add`, `multiply`).
- `Write.value_dtype` is the stored value's type. The store converts it to the buffer's element type.
- Integer arithmetic is always restamped from its typed operands.

A statement whose dtype is `None` takes its type from the renderer's inference. Only hand-built test bodies leave it
unset.

### 4.2 Fragment types

A fragment is a per-thread register array. Its type is `(role, shape, dtype, count)` from its `RegFragment`. The
register count per lane is `nregs` if given, else:

```
nregs(a, (m, n, k), t) = m·k / (32 · p(t))        p(t) = 4 if sizeof(t) = 1 else 2
nregs(b, (m, n, k), t) = k·n / (32 · p(t))
nregs(c, (m, n, k), t) = m·n / 64  if sizeof(t) = 2  else  m·n / 32
```

For `m16n8k16` at f16 this gives `a[4]`, `b[2]`, `c[4]` (f32) or `c[2]` (packed f16). A `c` fragment starts at zero.

A `FragLayout` gives the meaning of each C-fragment register. Element `i` of a lane sits at in-tile row
`row_off[elem_row[i]]` and in-atom column `col_off[i]`, both expressions over the layout's lane names:

| Layout | `n_elems` | element rows | element columns | row reduce masks |
|---|---|---|---|---|
| `M16N8` (`m16n8k16`, `wgmma`) | 4 | `g`, `g`, `g+8`, `g+8` with `g = lane/4` | `2t`, `2t+1`, `2t`, `2t+1` with `t = lane%4` | `2, 1` |
| `M8N8K4` (Volta, one logical 16×16) | 8 | `r, r, r+2, r+2, r, r, r+2, r+2` | `c, c+1, c, c+1, c+4, c+5, c+4, c+5` | `4, 2` |

### 4.3 Typing rules for fragment statements

```
MmaSyncPtx(c, a, b, (m,n,k), ab, cd):   a : (a,(m,n,k),ab)   b : (b,(m,n,k),ab)   c : (c,(m,n,k),cd)
WgmmaMma(cs, bd, (64,n,16), …):         8 ≤ n ≤ 256,  n % 8 = 0,  |cs| = n/8,  each cᵢ : (c,(16,8,16),f32)
                                         exactly one of a_desc, a_frag;  a_frag ⇒ trans_a = 0
FragmentPromote(dst, src):              src : packed f16 C fragment,  dst : f32 C fragment, same layout
FragmentApply(out, op, args, kinds):    |args| = |kinds| = arity(op);  in_place ⇒ out is a FRAG arg;
                                         any template reads __frow ⇒ row_base ≠ None
FragmentRowReduce(top, bot, frags, op): each frag : C fragment of `layout`;  top, bot : dtype
RegStore(dst, idx, frag, …):            frag, extra_frags : C fragments;  epilogue params bind them in order
```

## 5. Well-formedness

A `KernelOp` is well formed when every rule below holds. §9 says where each one is enforced.

**K1 — Root shape.** The top-level body is `ZeroPrologue* · Tile`: exactly one `Tile` at the top level, optionally
preceded by delegated zero-init statements. Everything else is nested inside the `Tile`.

**K2 — One launch geometry.** Only the root `Tile` binds grid coordinates.
`aux_threads > 0 ⇒ block_threads ≠ None`.
A grouped raster is active only when `raster_group > 1` and both `raster_axes` are adjacent in `axes`, `m` before `n`.

**K3 — Closed and scoped.** §3.2 and §3.3 hold.

**K4 — Index arity.** Every index into a buffer spells every dimension of that buffer, with `0` at a size-one
dimension. A single coordinate is an already flat address. The renderer flattens row-major and refuses any other count.

**K5 — Bounded loops.** `StridedLoop.end`, when set, never exceeds the axis extent. A `Loop` runs exactly
`axis.extent` times.

**K6 — Folds.** A `Loop` or `StridedLoop` is a reduce loop if and only if its body holds an `Accum`. All `Accum`s
with one `name` in one loop agree on `op`. A cross-thread combine requires a commutative op.

**K7 — Barriers.**
- `Sync(warp=True) ⇒ barrier_id = 0`.
- `barrier_id > 0 ⇒ count ≠ None`, `1 ≤ barrier_id ≤ 15`, and exactly `count` threads reach the barrier.
- A CTA-wide `Sync()` is reached by every thread of the CTA. Inside a branch that only some warps take, the barrier
  must be a named `bar.sync`. The same holds for `TreeHalve.barrier_id`.

**K8 — Shared memory.** `smem_bytes(op) ≤ ctx.max_dynamic_smem`, where `smem_bytes` packs the `Smem` buffers in body
order, each aligned to `max(sizeof(dtype), align)`. A swizzled TMA slab aligns to its swizzle atom
(`B128 → 1024`, `B64 → 512`, `B32 → 256` bytes).

**K9 — Asynchronous copies.**
- `CpAsyncCopy.nbytes ∈ {4, 8, 16}`. A 16-byte copy needs 16-byte-aligned source and destination.
- A thread reads a `cp.async` destination only after a `CpAsyncWait` that drains its group, followed by a `Sync`
  when other threads issued the copy.
- A `TmaLoad` is issued by one thread, after an `MbarrierArriveExpectTx` on the same barrier and slot. Every reader
  waits with `MbarrierWait` on that barrier and phase first.
- Each mbarrier slot is initialized once by `MbarrierInit`, in the prologue, by one thread, followed by `Sync`.
- `SetMaxNReg` and the TMA, mbarrier and `wgmma` statements need sm_90 or newer.

**K10 — Warp collectives.**
- `WarpShuffle.length` and `WarpBroadcast.length` are powers of two, at most 32. The warp is converged when it is
  reached.
- `TreeHalve.length` is a power of two, at most `blockDim.x`, and its `bufs` are filled before it runs.
- Every lane of a warp takes part in `LdmatrixLoad`, `MmaSyncPtx`, `FragmentRepack` and `FragmentRowReduce`. Every warp
  of a warp group takes part in `WgmmaMma`, `WgmmaFence`, `WgmmaCommit` and `WgmmaWait`.

**K11 — wgmma discipline.** A `WgmmaFence` comes before the first `WgmmaMma` after anything else touched its
accumulator. A `WgmmaCommit` closes each chunk of cells. A `WgmmaWait` comes before any read of those accumulators
and before the slot that fed them is released.

**K12 — Fragment use.** A fragment is declared by one `RegFragment` before its first use. `LdmatrixLoad.staged =
True` reads shared memory (except on the Volta layout, which gathers from either space). `gmem_guard` and `k_zero`
are set only when `staged = False`. `scale_buffer` and `BlockScaleLoad` never apply to the same operand.

**K13 — Paged buffers.** A buffer that is paged and spans more than one page is touched only by `Load` and `Write`.
It is never zero-initialized per launch.

**K14 — No decisions.** A `KernelOp` carries no scheduling decision. Its `knobs` hold only the on-by-default policy
knobs its peepholes record for idempotence (`VECTORIZE_LOADS`, `VECTORIZE_STORES`, `GUARD_REDUCTIONS`,
`INTERLEAVE_LOADS`, `PAIR_LDMATRIX`, `FAST_EXP`, …).

## 6. Execution model

### 6.1 Machine state

A launch runs `G` CTAs of `B` threads each (§7.2). The state is:

```
M = ( Gmem : buffer ↦ array,                                     -- global memory, shared by all CTAs
      Smem_c : local buffer ↦ array            for each CTA c,   -- shared memory, one per CTA
      Bar_c  : mbarrier slot ↦ (phase, pending arrivals, pending bytes),
      Async_t : queue of in-flight cp.async groups              for each thread t,
      Env_t  : name ↦ value                     for each thread t -- SSA values, axis values, fragment registers )
```

Every thread runs the same body. A thread's identity is `(blockIdx.x, threadIdx.x)`. Statements in one thread run
in program order. Threads interleave freely between barriers; Kernel IR gives no other ordering between threads.

### 6.2 The `Tile`

Let `e₀ … eₙ₋₁` be the extents of `axes` and `N = ∏ eᵢ`. Let `C = block_threads` for a cooperative tile and
`C = 256` otherwise. Each thread computes

```
gid = blockIdx.x · C + (threadIdx.x mod C)      if aux_threads > 0
gid = blockIdx.x · blockDim.x + threadIdx.x     otherwise
```

If `gid < N`, the thread binds the axes by the row-major decode

```
aᵢ = ⌊gid / sᵢ⌋ mod eᵢ        where sᵢ = ∏_{j>i} eⱼ
```

and runs `body` once. With an active grouped raster on `(m, n)`, the pair is decoded by the standard threadblock
swizzle instead: `G` tiles of the grouped axis run fastest inside each stripe, and the last group is clamped.

The guard `gid < N` is omitted when it is always true: `N mod C = 0` and no aux band, or `spread = True`. With
`spread = True` the tile is one cell and every thread runs the body; the free loops at its top stride over the CTA.

Threads with `threadIdx.x ≥ C` are the warp-specialized producer band. They decode the CTA's first cells, so their
block-axis variables are correct. Their per-cell variables alias a compute cell and the body never reads them on that
path.

The id is 32-bit when `N ≤ 2³¹ − 1` and static. It is 64-bit otherwise and always 64-bit for a symbolic grid.

### 6.3 Control

```
⟦Loop(a, body)⟧          = for a in 0 .. extent(a) − 1:  ⟦body⟧
⟦StridedLoop(a, s, d, body, end)⟧ = for (a = s; a < (end ?? extent(a)); a += d):  ⟦body⟧
⟦Cond(c, b, e)⟧          = if ⟦c⟧ then ⟦b⟧ else ⟦e⟧
```

`unroll` adds `#pragma unroll` and does not change meaning.

A reduce loop seeds each `Accum` target with `op.identity` before the first trip (when `seed = True`) and folds
inside:

```
⟦Accum(x, v, op)⟧ = Env[x] := op(Env[base ?? x], Env[v])
```

After the loop, `x` holds the folded value in the enclosing scope.

### 6.4 Scalars

```
⟦Assign(x, op, (y₁…yₖ), t)⟧  = Env[x] := cast_t(op(Env[y₁], …, Env[yₖ]))
⟦Let(x, e, t)⟧               = Env[x] := cast_t(⟦e⟧)
⟦Select(x, ((v₁,p₁)…(vₖ,pₖ)))⟧ = Env[x] := Env[vᵢ] for the first i with ⟦pᵢ⟧, else Env[vₖ]
⟦Init(x, c, t)⟧              = Env[x] := cast_t(c)
```

`exp` may run as `__expf` when `085_fast_exp` recorded it. Division by a loop-invariant divisor may run as a multiply
by a hoisted reciprocal under `FAST_MATH`. Both are the only deviations from the numpy meaning of `op`.

### 6.5 Memory

```
⟦Load((x₀…xₖ₋₁), g, i)⟧   = Env[xⱼ] := Mem(g)[flat(g, i) + j]                    for j < k
⟦Write(g, i, (v₀…vₖ₋₁))⟧  = Mem(g)[flat(g, swz(i)) + j] := Env[vⱼ]               for j < k
                            atomic:  Mem(g)[…] += Env[vⱼ]
⟦ZeroPrologue(g, w)⟧      = the grid zeroes words 0 … w−1 of g, each thread a stride
⟦Carry(x, v, i, seed)⟧    = the state x at cell i takes Env[v] for the next step
⟦Pre(x, c, i)⟧            = Env[x] := the previous step's state c at cell i (seed on the first step)
⟦Smem(s, ext, t, al)⟧     = reserve a shared array of ∏ext elements of t in this CTA
```

`Mem(g)` is `Gmem` for a graph buffer and `Smem_c` for a local buffer. `flat(g, i)` is the row-major offset of `i`
in `g`'s shape. A paged buffer resolves its page from the index along the paged axis first. A `literal_constants`
input is not read: its `Load` binds the constant.

A `Load` with `carried` keeps a staged copy live across a loop's back edge: `"load"` refills only the packed vector,
`"unpack"` re-binds the names from it and touches no memory.

### 6.6 Barriers and asynchronous copies

```
⟦Sync()⟧                 = every thread of the CTA arrives; shared and global writes before it are visible after it
⟦Sync(warp=True)⟧        = the same, for the threads of one warp
⟦Sync(id, n)⟧            = the same, for the n threads arriving on named barrier id
⟦CpAsyncCopy(s, i, g, j, n)⟧ = enqueue on Async_t: Smem_c[s][swz(i) … +n bytes] := Gmem[g][j … +n bytes]
                               valid = false:  the destination bytes become zero and nothing is read
⟦CpAsyncCommit⟧          = close the copies enqueued since the last commit into one group
⟦CpAsyncWait(k)⟧         = block until at most k of this thread's groups are in flight
⟦MbarrierInit(m, n, sl)⟧ = Bar_c[m, sl] := (phase 0, n expected arrivals)
⟦MbarrierArriveExpectTx(m, b, sl)⟧ = arrive on Bar_c[m, sl] and expect b more bytes
⟦TmaLoad(s, i, d, x, m, sl)⟧ = copy the box at coordinates x of descriptor d into Smem_c[s] at i;
                               complete its bytes on Bar_c[m, sl]
⟦MbarrierArrive(m, sl)⟧  = arrive on Bar_c[m, sl]
⟦MbarrierWait(m, p, sl)⟧ = block until Bar_c[m, sl] has completed phase p
⟦SetMaxNReg(n, dir)⟧     = raise or lower this warp's register budget to n
```

A phase of an mbarrier completes when its expected arrivals and its expected bytes have both arrived. Its
completion makes the TMA bytes visible to every thread that waited on it.

### 6.7 Cross-thread combines

Let `⊕` be the merge given by `combine_states`: it reads `state` and `state_b` and writes `state`.

```
⟦WarpShuffle(st, stb, ⊕, L, prefix=False)⟧ =
    for s in L/2, L/4, …, 1:  stb := shfl_xor(st, s);  st := st ⊕ stb
    -- afterwards every lane of each aligned L-lane group holds ⊕ over the group

⟦WarpShuffle(st, stb, ⊕, L, prefix=True)⟧ =
    for s in 1, 2, …, L/2:    stb := shfl_up(st, s);  if lane mod L ≥ s: st := st ⊕ stb
    -- afterwards lane j of each group holds the inclusive prefix ⊕ over lanes 0 … j

⟦WarpBroadcast(st, L)⟧ = st := st of the last lane of the aligned L-lane group

⟦TreeHalve(bufs, st, stb, ⊕, L, tid)⟧ =
    for s in L/2, L/4, …, 1:  if tid < s: bufs[tid] := bufs[tid] ⊕ bufs[tid + s];  barrier
    st := bufs[0]                    -- every thread ends with the full reduction, in place
```

`⊕` is commutative (K6), so the operand order of a merge does not matter. `inner = (v, w)` makes each slab
hold `L` segments of `w` slots and halve the segment index at slot `v`.

### 6.8 Tensor-core fragments

```
⟦RegFragment(f, c, …)⟧            = declare f; a C fragment starts at 0
⟦LdmatrixLoad(f, src, i, role, …)⟧ = fill f with the operand tile at src[i] in the layout's lane map
                                     (shared: ldmatrix x2/x4, .trans for b_trans; global: a gather;
                                      guards clamp lanes past the bound, k_zero zeroes K past it;
                                      pair_frag: one x4 fills f and pair_frag)
⟦BlockScaleLoad(f, src, i, role)⟧  = the supplying lanes load one b32 of four ue4m3 scales into f
⟦MmaSyncPtx(c, a, b, …)⟧           = c := a · b + c       on the warp's tile, in the layout's lane map
                                     block-scaled: a and b are scaled by sfa, sfb per 16 K elements
⟦FragmentPromote(d, s)⟧            = d := d + f32(s);  s := 0
⟦FragmentRepack(f, (s₁…), …)⟧      = f := the 16-bit A or B operand whose values are those of the C fragments sᵢ
⟦FragmentApply(o, op, args, kinds)⟧ = for each element e of the layout:
                                       o[e] := post(op(arg₁(e), …, argₖ(e)))
      where  FRAG: x[e]   ROW: x₀ if row(e) is g else x₁   UNIFORM: x
             COORD: ⟦p⟧ with __frow := row_base + row(e), __fcol := col_base + col(e)
             GMEM:  f32(g[i]) at those coordinates
⟦FragmentRowReduce(t, b, fs, op)⟧  = t, b := op over every element of fs in row g, g+8;
                                     then butterfly over the layout's reduce_xors
⟦RegStore(g, i, f, …)⟧             = for each element e: if guards hold at e:
                                       g[i + (row(e), col(e))] := conv(epilogue(f[e], extra[e], …))   (atomic: +=)
```

### 6.9 wgmma

```
⟦WgmmaDescriptor(d, s, i, swz, lbo, sbo)⟧ = d := the 64-bit descriptor of slab s at element i
⟦WgmmaFence⟧                     = order earlier register and shared accesses before the next WgmmaMma
⟦WgmmaMma(cs, bd, (64,n,16), …)⟧ = issue, asynchronously:  cs := A · B + (scale_d ? cs : 0)
                                   A from a_desc or a_frag, B from bd; d[4j+k] is cs[j][k]
⟦WgmmaCommit⟧                    = close the cells issued since the last commit into one group
⟦WgmmaWait(k)⟧                   = block until at most k committed groups are in flight
```

An accumulator written by a `WgmmaMma` has an unspecified value until a `WgmmaWait` covers its group (K11).

## 7. Signature and launch

### 7.1 Signature

The kernel's parameters are derived from the body, in this order:

1. one `const T* __restrict__` per input buffer — `Load.input`, `CpAsyncCopy.src`, `TmaDescriptor.src_buf` and the
   other external reads — in first-appearance order, skipping `literal_constants`;
2. one `T* __restrict__` per output buffer — `Write.output`, `RegStore.dst_buffer`, `ZeroPrologue.dst` — in
   first-appearance order;
3. one `const CUtensorMap*` per distinct `TmaDescriptor.name`;
4. one `int` per runtime argument: each symbolic size name the body reads, then each `serial` axis.

A paged buffer takes a page table `T* const* <name>__pages` in place of its pointer. `Smem` buffers are never
parameters. Every buffer name must be a C identifier. `KernelOp.inputs` and `outputs` hold the same lists, in the
same order, as graph tensors.

### 7.2 Launch

```
C          = block_threads ?? 256
blockDim.x = C + aux_threads
gridDim.x  = ⌈N / C⌉                      (a runtime expression when N is symbolic)
launches   = one per coordinate of the serial axes, in order
dyn. smem  = smem_bytes(op)               when it exceeds 48 KiB; otherwise 0
__launch_bounds__(C + aux_threads)
```

Over 48 KiB, every `Smem` becomes an alias into one `extern __shared__` pool at its packed offset, aligned to the
largest buffer alignment and at least 16 bytes.

## 8. Rendering

`render_kernelop` emits, in order: the `#include`s and inline-PTX preludes the body needs (fp8 decode, mma.sync,
wgmma wrappers, cp.async, swizzle helpers, TMA), the signature, `griddepcontrol.wait`, the dynamic pool, the `lane`
and `warp` ids when used, and then each statement's own `render`.

| Statement | CUDA |
|---|---|
| `Tile` | `_gid` decode, optional `if (_gid < N)` guard, axis variables |
| `Loop` / `StridedLoop` | `for` loop, with `#pragma unroll` when `unroll` |
| `Cond` | `if (…) { … } else { … }` |
| `Assign`, `Let`, `Select`, `Init` | one typed local declaration |
| `Accum` | `x = op(x, v);` with the seed declared before the loop |
| `Load` / `Write` | scalar or vector access at the flattened index; `atomicAdd` when atomic |
| `Smem` | `__shared__ [__align__(a)] T s[n];` or a pool alias |
| `Sync` | `__syncthreads()`, `__syncwarp()` or `bar.sync id, n` |
| `CpAsync*` | `cp.async.{ca,cg}.shared.global`, `cp.async.commit_group`, `cp.async.wait_group` |
| `TmaDescriptor` | nothing in the body; the host encodes it with `cuTensorMapEncodeTiled` |
| `TmaLoad`, `Mbarrier*` | `cp.async.bulk.tensor.<r>d`, `mbarrier.{init,arrive,arrive.expect_tx,try_wait.parity}` |
| `SetMaxNReg` | `setmaxnreg.{inc,dec}.sync.aligned.u32` |
| `WarpShuffle` / `WarpBroadcast` | `__shfl_xor_sync` / `__shfl_up_sync` / `__shfl_sync` over `__activemask()` |
| `TreeHalve` | a halving loop over the shared slabs with a barrier per step |
| `RegFragment` | `float c[n]` / `unsigned a[n]`, arrayed when `count > 1` |
| `LdmatrixLoad`, `MmaSyncPtx`, `FragmentRepack`, `FragmentPromote` | calls to the `emmy_mma_*` / `emmy_ldmatrix_*` prelude wrappers |
| `FragmentApply` | `#pragma unroll for (_e)` over the elements |
| `RegStore` | per-lane packed or scalar stores with the epilogue inlined |
| `Wgmma*` | `wgmma.fence`, `wgmma.mma_async`, `wgmma.commit_group`, `wgmma.wait_group` |

The printed form (`--ir kernel`, `pretty()`) is a readable listing of the same tree. A small cooperative row
reduction prints as:

```
Tile[m, lane_k] (N=4096)
  for k in lane_k..1024:32
    x = load X[m, k]
    acc <- add(acc, x)
  WarpShuffle(acc, length=32, prefix=False)
  if (lane_k == 0):
    Y[m] = acc
```

## 9. Where the rules are enforced

| Rule | Enforced by |
|---|---|
| field constraints of `WgmmaMma`, `Tile.aux_threads`, `Select`, `Affine`-free body | constructors (`__post_init__`) and `010_materialize` |
| K1 one top-level `Tile` | `lowering/cuda/010_lower_kernelop` refuses any other shape |
| K3 closed body | `010_materialize._unbound_names` assertion |
| K3 duplicate binding | `010_materialize._drop_repeated_declarations` drops identical repeats; others reach nvcc |
| K4 index arity | `render_index` refuses a count that is neither 1 nor the buffer's rank |
| K7 named barrier needs `count`, warp sync has no id | `Sync.render` |
| K8 shared memory | `KernelOp.validate` against `max_dynamic_smem` |
| K13 paged buffers | `010_lower_kernelop` |
| buffer names are C identifiers | `render_kernelop` |
| unknown fragment layout | `frag_layout` raises |
| K2, K5, K6, K9 – K12, K14 | contracts of `lowering/kernel`; not checked on the IR. A violation shows up as a CUDA compile error, a launch error or a wrong result |

`KernelOp.validate` also names two launch checks — threads per CTA and the CTA cap of 65 536 — that are not active
today; only the shared-memory check runs.
