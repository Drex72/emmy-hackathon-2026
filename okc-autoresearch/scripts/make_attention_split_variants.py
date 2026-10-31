#!/usr/bin/env python3
from pathlib import Path
import argparse, re
ENTRY_RE = re.compile(r"_TUNED\s*=\s*\{", re.M)
def patch(src, d1, v1):
    m=ENTRY_RE.search(src)
    if not m: raise SystemExit("could not find _TUNED table")
    ins=[]
    if d1 is not None: ins.append(f"    (1, 1, True, 1): dict(n_splits={d1}, nc=1, coop=0),")
    if v1 is not None: ins.append(f"    (1, 8, True, 1): dict(n_splits={v1}, nc=1, coop=1),")
    return src[:m.end()]+"\n"+"\n".join(ins)+src[m.end():]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("outdir"); a=ap.parse_args()
    src=Path(a.source).read_text(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    for name,d1,v1 in [("A06-d1-16",16,None),("A06-v1-16",None,16),("A06-d1-8",8,None),("A06-v1-8",None,8),("A06-both-16",16,16)]:
        (out/(name+".py")).write_text(patch(src,d1,v1))
if __name__=="__main__": main()
