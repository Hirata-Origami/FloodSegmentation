"""Free analysis on saved probabilities (no GPU): official-split ensembles across variants, pre-specified composition.
Uses jobs_results/final__*.npz written by notebooks 04-06. Threshold is chosen on valid only; nothing is tuned on test.
Usage: python scripts/cross_variant_ensemble.py --cache /kaggle/working/cache --results /kaggle/working/out/jobs_results
"""
import argparse, glob, json, os
import numpy as np

def vid(v): return v.replace("+", "p").replace("|", "_").replace(":", "_")
def counts(p, y, t):
    m = y >= 0; pr = (p >= t) & m; tt = (y == 1) & m
    return np.stack([(pr & tt).sum((1, 2)), (pr & ~tt).sum((1, 2)), (~pr & tt).sum((1, 2))], 1)
def iou(c): return float(c[:, 0].sum() / max(c.sum(), 1))
def boot(ca, cb=None, n=2000, seed=0):
    rng = np.random.default_rng(seed); k = len(ca); a, d = [], []
    for _ in range(n):
        i = rng.integers(0, k, k); a.append(iou(ca[i]))
        if cb is not None: d.append(iou(ca[i]) - iou(cb[i]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(a) if cb is None else (float(np.mean(d)), q(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True); ap.add_argument("--results", required=True)
    a = ap.parse_args(); Y = np.load(f"{a.cache}/Y.npy"); meta = json.load(open(f"{a.cache}/meta.json")); I = {k: np.array(v) for k, v in meta["idx"].items()}
    VARS = ["S1", "S1+ref", "S1+dw", "S1+ref+dw"]; P = {}
    for v in VARS:
        fs = sorted(glob.glob(f"{a.results}/final__{vid(v)}__*.npz"))
        if fs:
            ds = [np.load(f) for f in fs]; P[v] = {s: np.mean([d[s].astype(np.float32) for d in ds], 0) for s in ("valid", "test", "bol")}; print(f"{v}: {len(fs)} models")
    combos = {"S1": ["S1"], "S1+ref": ["S1+ref"], "S1+dw": ["S1+dw"], "S1+ref+dw": ["S1+ref+dw"],
              "cross-variant (ref, dw, ref+dw)": ["S1+ref", "S1+dw", "S1+ref+dw"]}      # composition fixed in advance
    res = {}
    for name, vs in combos.items():
        if not all(v in P for v in vs): continue
        pv = np.mean([P[v]["valid"] for v in vs], 0); grid = np.arange(0.30, 0.71, 0.02)
        thr = float(grid[int(np.argmax([iou(counts(pv, Y[I["valid"]], t)) for t in grid]))])
        ct = counts(np.mean([P[v]["test"] for v in vs], 0), Y[I["test"]], thr); cb = counts(np.mean([P[v]["bol"] for v in vs], 0), Y[I["bol"]], thr)
        res[name] = ct; lo, hi = boot(ct); print(f"{name:34s} thr {thr:.2f} | test {iou(ct):.4f} ({lo:.3f}-{hi:.3f}) | Bolivia {iou(cb):.4f}")
    for name, ct in res.items():
        if name != "S1" and "S1" in res:
            d, (lo, hi) = boot(ct, res["S1"]); print(f"paired test diff {name} minus S1: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__": main()
