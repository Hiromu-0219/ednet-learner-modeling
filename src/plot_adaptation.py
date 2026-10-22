# 適応曲線の図（data/processed/adaptation/adaptation.csv から）。
# 前半をすべて使った点（探索的、事前登録の外）は、reports/adaptation.md の値を引数で渡して重ねる。
# 使い方：python src/plot_adaptation.py --full-half-k 1942 --full-half-diff -0.0326 --full-half-lo -0.0489 --full-half-hi -0.0198
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
for a in ["--full-half-k", "--full-half-diff", "--full-half-lo", "--full-half-hi"]:
    ap.add_argument(a, type=float)
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / "data/processed/adaptation/adaptation.csv")
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
K = list(r["k"].drop_duplicates())
pos = {k: i for i, k in enumerate(K)}                      # k を等間隔に並べる
FULL_X = len(K) + 0.6
t = r[r["評価"].eq("決定点（主）")]
ax = axes[0]
for m, color, style, label in [("M1", MUTED, "--", "M1 (population, no adaptation)"), ("M2", C2, "-", "M2 (learner only)"),
                               ("M3", C1, "-", "M3 (hierarchical)")]:
    ax.plot([pos[k] for k in t["k"]], t[f"{m}_log_loss"], style, color=color, marker="o" if m != "M1" else None,
            markersize=5, linewidth=2, label=label)
ax.set_xticks(range(len(K)), [str(k) for k in K])
ax.set_ylim(0.5, 0.9)
ax.set_xlabel("first k events of an unseen learner")
ax.set_ylabel("log loss at decision points")
ax.set_title("Decision points, 269 unseen learners\n(M2 at k=0-50 is above 0.9, off scale)", loc="left", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = axes[1]
for ev, color, label, off in [("決定点（主）", C1, "decision points (main)", -0.08), ("全イベント（副）", C2, "all events (secondary)", 0.08)]:
    t = r[r["評価"].eq(ev)]
    x = [pos[k] + off for k in t["k"]]
    ax.errorbar(x, t["M3-M1_log_loss"], yerr=[t["M3-M1_log_loss"] - t["M3-M1_log_loss_lo"], t["M3-M1_log_loss_hi"] - t["M3-M1_log_loss"]],
                fmt="o-", color=color, ecolor=color, elinewidth=1.5, capsize=0, markersize=5, linewidth=1.5, label=label)
if args.full_half_k:
    ax.errorbar([FULL_X], [args.full_half_diff], yerr=[[args.full_half_diff - args.full_half_lo], [args.full_half_hi - args.full_half_diff]],
                fmt="D", color=C1, mfc="white", ecolor=C1, elinewidth=1.5, capsize=0, markersize=6,
                label="whole first half (exploratory)")
ax.axhline(0, color=INK, linewidth=1)
ax.set_xticks(list(range(len(K))) + [FULL_X], [str(k) for k in K] + [f"all\n(~{int(args.full_half_k or 0):,})"])
ax.set_xlim(-0.5, FULL_X + 0.5)
ax.set_xlabel("first k events of an unseen learner")
ax.set_ylabel("M3 - M1 log loss (below 0: M3 better)")
ax.set_title("Gain from learner-specific estimates\n95% learner bootstrap CI", loc="left", fontsize=9, color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout()
out = ROOT / "reports/figures/adaptation_curve.png"
fig.savefig(out, dpi=130)
print("wrote", out.relative_to(ROOT))
