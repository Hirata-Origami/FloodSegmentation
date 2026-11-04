"""Sample prediction images from a finished notebook-06 run (run on Kaggle or after downloading the output).
For each chosen held-out event it picks the chips where the pre-event/land-cover model removed the most false positives
relative to S1 and plots: VV, VH, label, S1 error map, new-model error map.
Error colours: green = true positive, red = false positive, blue = false negative, grey = true negative, black = ignored.

Usage:
  python scripts/make_qualitative_figures.py --cache /kaggle/working/cache --results /kaggle/working/out/jobs_results \
      --events Somalia Pakistan Bolivia --base S1 --new S1+ref+dw --seed 0 --per-event 2 --out figures
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def vid(v): return v.replace("+", "p").replace("|", "_").replace(":", "_")      # same sanitising as the notebooks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--events", nargs="+", default=["Somalia", "Pakistan", "Bolivia"])
    ap.add_argument("--base", default="S1"); ap.add_argument("--new", default="S1+ref+dw")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--per-event", type=int, default=2)
    ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--out", default="figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    X = np.load(f"{a.cache}/X.npy", mmap_mode="r"); Y = np.load(f"{a.cache}/Y.npy"); meta = json.load(open(f"{a.cache}/meta.json"))
    EV = np.array(meta["event"]); ids = meta["ids"]
    cmap = ListedColormap(["black", "#bdbdbd", "#2ca02c", "#d62728", "#1f77b4"])      # 0 ignored, 1 TN, 2 TP, 3 FP, 4 FN
    def err_map(prob, y):
        pr = prob >= a.thr; out = np.ones(y.shape, np.uint8)
        out[(pr) & (y == 1)] = 2; out[(pr) & (y == 0)] = 3; out[(~pr) & (y == 1)] = 4; out[y < 0] = 0
        return out
    def stretch(im):
        lo, hi = np.percentile(im, [2, 98]); return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)
    for e in a.events:
        pb = np.load(f"{a.results}/loeo__{vid(a.base)}__{e}__s{a.seed}.npz"); pn = np.load(f"{a.results}/loeo__{vid(a.new)}__{e}__s{a.seed}.npz")
        assert (pb["idx"] == pn["idx"]).all()
        idx = pb["idx"]; fp_b = [((pb["p"][k] >= a.thr) & (Y[i] == 0)).sum() for k, i in enumerate(idx)]
        fp_n = [((pn["p"][k] >= a.thr) & (Y[i] == 0)).sum() for k, i in enumerate(idx)]
        order = np.argsort(np.array(fp_n) - np.array(fp_b))[:a.per_event]          # largest false-positive reduction first
        fig, axs = plt.subplots(len(order), 5, figsize=(14, 2.9 * len(order)), squeeze=False)
        for r, k in enumerate(order):
            i = idx[k]; vv = np.asarray(X[i][0], np.float32); vh = np.asarray(X[i][1], np.float32)
            panels = [(stretch(vv), "gray", f"VV (normalised)\n{ids[i]}"), (stretch(vh), "gray", "VH (normalised)"),
                      (np.where(Y[i] < 0, 0, np.where(Y[i] == 1, 2, 1)), cmap, "label (green = water)"),
                      (err_map(pb["p"][k], Y[i]), cmap, f"{a.base}: FP {fp_b[k]}"), (err_map(pn["p"][k], Y[i]), cmap, f"{a.new}: FP {fp_n[k]}")]
            for c, (img, cm, title) in enumerate(panels):
                axs[r, c].imshow(img, cmap=cm, vmin=0, vmax=4 if cm is cmap else 1, interpolation="nearest"); axs[r, c].set_title(title, fontsize=8); axs[r, c].axis("off")
        fig.suptitle(f"Held-out event: {e} (leave-one-event-out, seed {a.seed}, threshold {a.thr}); green TP, red FP, blue FN", fontsize=10)
        plt.tight_layout(); fn = f"{a.out}/qualitative_{e}.png"; plt.savefig(fn, dpi=110); plt.close(); print("wrote", fn)

if __name__ == "__main__": main()
