"""Summary figures from the result CSVs in results/ (no data or GPU needed).
Usage: python scripts/make_summary_figures.py   (run from the repository root)"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
C = {"S1": "#7f7f7f", "S1+ref": "#1f77b4", "S1+dw": "#2ca02c", "S1+ref+dw": "#d62728"}
def rd(p): return list(csv.DictReader(open(p)))
f = lambda x: float(x) if x not in ("", None) else np.nan

# ---- fig 1: 3-seed LOEO macro IoU per variant
S = rd("results/loeo_3seed_summary.csv"); V = [r["variant"] for r in S]
fig, ax = plt.subplots(figsize=(6.4, 4))
for i, r in enumerate(S):
    seeds = [f(r["seed0_macro_IoU"]), f(r["seed1_macro_IoU"]), f(r["seed2_macro_IoU"])]
    ax.bar(i, f(r["seed_mean"]), color=C[r["variant"]], alpha=.55, width=.6)
    ax.scatter([i] * 3, seeds, color="k", zorder=3, s=18)
    ax.text(i, 0.545, f"{f(r['seed_mean']):.3f}", ha="center", fontsize=9)
    if r["paired_diff_vs_S1_seed_averaged"]:
        ax.text(i, 0.625, f"+{f(r['paired_diff_vs_S1_seed_averaged'])+1e-9:.3f}\n({f(r['ci95_low']):+.3f}, {f(r['ci95_high']):+.3f})", ha="center", fontsize=8)
ax.set_xticks(range(len(V))); ax.set_xticklabels(V); ax.set_ylim(0.54, 0.66)
ax.set_ylabel("LOEO macro IoU (11 events)"); ax.set_title("Leave-one-event-out, 3 seeds (dots = seeds)\nlabels: paired diff vs S1 (95% CI)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig1_loeo_3seed_macro_iou.png", dpi=130); plt.close()

# ---- fig 2: per-event IoU (seed-ensemble)
P = rd("results/loeo_3seed_per_event_iou_seed_ensemble.csv"); ev = [k for k in P[0] if k != "variant"]
base = {r["variant"]: r for r in P}["S1"]; order = sorted(ev, key=lambda e: f(base[e]))
fig, ax = plt.subplots(figsize=(10, 4)); w = .2
for j, r in enumerate(P):
    ax.bar(np.arange(len(order)) + (j - 1.5) * w, [f(r[e]) for e in order], w, label=r["variant"], color=C[r["variant"]])
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30); ax.set_ylabel("water IoU (held-out event)")
ax.legend(ncol=4, fontsize=8, loc="upper left"); ax.set_title("Per-event IoU, leave-one-event-out (3-seed ensemble); events sorted by S1 IoU", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig2_loeo_per_event_iou.png", dpi=130); plt.close()

# ---- fig 3: precision on hard events + matched-recall precision
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.arange(3); w = .2
for j, r in enumerate(S):
    vals = [f(r["seed_ensemble_macro_precision"]), f(r["Somalia_precision"]), f(r["Pakistan_precision"])]
    axs[0].bar(x + (j - 1.5) * w, vals, w, label=r["variant"], color=C[r["variant"]])
axs[0].set_xticks(x); axs[0].set_xticklabels(["macro (11 events)", "Somalia", "Pakistan"]); axs[0].set_ylabel("precision at threshold 0.5")
axs[0].legend(fontsize=8); axs[0].set_title("Precision, seed-ensembled LOEO predictions", fontsize=10)
mm = {"S1": 0.706, "S1+ref": 0.768, "S1+dw": 0.750, "S1+ref+dw": 0.791}
axs[1].bar(list(mm), list(mm.values()), color=[C[k] for k in mm])
for i, (k, v) in enumerate(mm.items()): axs[1].text(i, v + .005, f"{v:.3f}", ha="center", fontsize=9)
axs[1].set_ylim(0.6, 0.83); axs[1].set_title("Macro precision at matched macro recall (0.744)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig3_precision.png", dpi=130); plt.close()

# ---- fig 4: everything tried, paired macro-IoU difference vs S1 (LOEO) with 95% CI
rows = [("Lovasz loss (1 seed)", .0124, -.0006, .0260), ("Tversky loss (1 seed)", -.0301, -.0524, -.0051),
        ("relative backscatter (1 seed)", -.0148, -.0270, -.0026), ("weather scalars (1 seed)", -.0615, -.0865, -.0319),
        ("all inputs incl. weather (1 seed)", -.0097, -.0287, .0099), ("terrain: HAND+slope (1 seed)", .006, -.004, .018),
        ("terrain: JRC+DEM+slope+HAND (1 seed)", .011, -.004, .027), ("rel+ref+dw (1 seed)", .0186, -.0027, .0407),
        ("land cover dw (3 seeds)", .0215, .0006, .0429), ("pre-event ref (3 seeds)", .0374, .0189, .0577), ("ref + dw (3 seeds)", .0420, .0203, .0638)]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for i, (n, d, lo, hi) in enumerate(rows):
    col = "#d62728" if lo > 0 else ("#7f7f7f" if hi > 0 and lo < 0 else "#1f77b4")
    ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col, capsize=3)
ax.axvline(0, color="k", lw=.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("paired LOEO macro-IoU difference vs S1 (95% CI)"); ax.set_title("Everything tried vs the single-date S1 baseline\n(red: CI above 0, grey: includes 0, blue: below 0)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig4_everything_tried.png", dpi=130); plt.close()

# ---- fig 5: official split test IoU with CI (15-model ensembles = final; 9-model earlier run; train+valid)
O = rd("results/official_split_15models.csv"); O9 = rd("results/official_split_ensembles.csv"); TV = rd("results/train_plus_valid_final.csv")
fig, ax = plt.subplots(figsize=(8, 4.4))
for i, v in enumerate(["S1", "S1+ref", "S1+dw", "S1+ref+dw"]):
    a_ = {r["variant"]: r for r in O}[v]; b_ = {r["variant"]: r for r in O9}[v]
    ax.errorbar(f(a_["test_IoU"]), i + .15, xerr=[[f(a_["test_IoU"]) - f(a_["test_ci95_low"])], [f(a_["test_ci95_high"]) - f(a_["test_IoU"])]], fmt="o", color=C[v], capsize=3)
    ax.errorbar(f(b_["test_IoU"]), i - .15, xerr=[[f(b_["test_IoU"]) - f(b_["test_ci95_low"])], [f(b_["test_ci95_high"]) - f(b_["test_IoU"])]], fmt="s", color=C[v], capsize=3, mfc="white")
for i, v in enumerate(["S1", "S1+ref+dw"]):
    t = {r["variant"]: r for r in TV}[v]
    ax.errorbar(f(t["test_IoU"]), 4.3 + i * .3, xerr=[[f(t["test_IoU"]) - f(t["test_ci95_low"])], [f(t["test_ci95_high"]) - f(t["test_IoU"])]], fmt="^", color=C[v], capsize=3)
ax.axvline(0.72, color="k", ls="--", lw=.8); ax.text(0.7215, 4.95, "0.72: DeepSARFlood reported\n(different inputs/protocol)", fontsize=7, va="center")
ax.set_yticks(list(range(4)) + [4.45]); ax.set_yticklabels(["S1", "S1+ref", "S1+dw", "S1+ref+dw", "train+valid fit\n(S1, ref+dw)"], fontsize=8)
ax.set_xlabel("test IoU with 95% bootstrap CI (90 chips)"); ax.set_xlim(0.55, 0.82); ax.set_ylim(-0.6, 5.3)
ax.set_title("Official split: circles = final 15-model ensembles, hollow squares = earlier 9-model run,\ntriangles = train+valid fit (non-standard protocol)", fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig5_official_split.png", dpi=130); plt.close()

# ---- fig 6: the 15 single models per variant (test IoU and Bolivia IoU)
SM = rd("results/official_split_single_models_15x4.csv"); mk = {"unet:resnet34": "o", "unet:resnext50_32x4d": "s", "unet:mit_b2": "^"}
fig, axs = plt.subplots(1, 2, figsize=(10, 4)); rng = np.random.default_rng(0)
for k, (col, ttl) in enumerate([("test_iou", "test IoU (90 chips)"), ("bolivia_iou", "Bolivia IoU (15 chips, held-out event)")]):
    for i, v in enumerate(["S1", "S1+ref", "S1+dw", "S1+ref+dw"]):
        for r in [r for r in SM if r["variant"] == v]:
            axs[k].scatter(i + rng.uniform(-.18, .18), f(r[col]), marker=mk[r["encoder"]], color=C[v], alpha=.8, s=28)
        m = np.mean([f(r[col]) for r in SM if r["variant"] == v]); axs[k].hlines(m, i - .3, i + .3, color="k", lw=1)
    axs[k].set_xticks(range(4)); axs[k].set_xticklabels(["S1", "S1+ref", "S1+dw", "S1+ref+dw"]); axs[k].set_title("Single models (15 per variant): " + ttl, fontsize=9)
axs[0].set_ylabel("IoU"); axs[1].annotate("S1+dw with mit_b2 collapses\non Bolivia (0.24-0.65)", (2, 0.25), (0.2, 0.3), fontsize=7, arrowprops=dict(arrowstyle="->"))
from matplotlib.lines import Line2D
axs[0].legend([Line2D([], [], marker=m, color="grey", ls="") for m in mk.values()], ["resnet34", "resnext50", "mit_b2"], fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("figures/fig6_single_models.png", dpi=130); plt.close()

# ---- fig 0: method / protocol overview
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, fc="#e8eef7", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#555")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#333"))
box(0.1, 3.2, 2.5, 1.0, "Sentinel-1 event image\nVV, VH (dB), VV-VH")
box(0.1, 1.9, 2.5, 1.0, "Pre-event Sentinel-1, same orbit\nyear median + std -> difference,\nz-score (ref)", "#fde9d9")
box(0.1, 0.6, 2.5, 1.0, "Dynamic World land-cover\nprobabilities, pre-event\nyear mean (dw)", "#e2f0d9")
box(3.6, 1.9, 2.1, 1.3, "U-Net\n(resnet34 / resnext50 /\nmit_b2 encoders)", "#fff2cc")
box(6.5, 1.9, 1.8, 1.3, "water mask\n(threshold from valid)")
for y in (3.7, 2.4, 1.1): arr(2.7, y, 3.55, 2.55)
arr(5.75, 2.55, 6.45, 2.55)
box(8.8, 3.1, 2.0, 1.3, "Protocol A: official split\ntrain 252 / valid 89 /\ntest 90 / Bolivia 15", "#ededed")
box(8.8, 1.4, 2.0, 1.5, "Protocol B: leave-one-event-out\n11 events, hold out one,\ntrain on the other 10", "#ededed")
arr(8.35, 2.7, 8.75, 3.6); arr(8.35, 2.4, 8.75, 2.1)
ax.text(5.5, 0.35, "Metrics: pooled water IoU, macro IoU over events, precision/recall; paired bootstrap CIs; matched-recall comparisons", ha="center", fontsize=8)
ax.set_title("Overview: inputs, model, and the two evaluation protocols", fontsize=11)
plt.tight_layout(); plt.savefig("figures/fig0_overview.png", dpi=130); plt.close()
print("figures written:", sorted(os.listdir("figures")))
