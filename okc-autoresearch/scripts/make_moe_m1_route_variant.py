#!/usr/bin/env python3
from pathlib import Path
import sys
HELPER = r'''
// M06: M=1 routing specialization. The 8 top-k global expert IDs are distinct,
// so every local expert count is 0 or 1. Preserve downstream tiny expert math.
__device__ __forceinline__ void route_m1(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  if (warp == 0) {
    if (lane < 8) {
      const int e = __ldg(p.topk_ids + lane) - p.lo;
      S.eid[lane] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
      S.pos[lane] = -1;
    }
    __syncwarp();
    int slot = -1;
#pragma unroll
    for (int i = 0; i < 8; ++i) if ((int)S.eid[i] == lane) slot = i;
    const unsigned active = __ballot_sync(0xffffffffu, slot >= 0 && lane < p.nloc);
    const unsigned below = active & ((lane == 0) ? 0u : ((1u << lane) - 1u));
    const int rank = __popc(below);
    const int nlocal = __popc(active);
    if (lane < p.nloc) {
      S.cnt[lane] = slot >= 0 ? 1 : 0;
      S.off[lane] = rank;
      if (slot >= 0) {
        S.jE[rank] = lane; S.jS[rank] = rank; S.jC[rank] = 1;
        S.ent[rank] = (short)slot; S.pos[slot] = (short)rank;
      }
    }
    if (lane == 0) { S.njF = 0; S.njS = nlocal; }
    __syncwarp();
  }
  CTA_SYNC();
  early_issue(p, S);
  CTA_SYNC();
}
'''
ANCHOR="__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {"
REPL="__device__ void early_issue(const Params& p, Smem& S);\n"+HELPER+"\n__device__ void route(const Params& p, Smem& S) {\n  if (p.M == 1) { route_m1(p, S); return; }"
def main():
    if len(sys.argv)!=3: raise SystemExit("usage: make_moe_m1_route_variant.py SOURCE DEST")
    src,dst=map(Path,sys.argv[1:]); t=src.read_text()
    if t.count(ANCHOR)!=1: raise SystemExit("routing anchor missing/ambiguous")
    dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(t.replace(ANCHOR,REPL,1))
if __name__=="__main__": main()
