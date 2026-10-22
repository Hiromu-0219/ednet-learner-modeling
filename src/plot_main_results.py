# 本番の評価結果（data/processed/main_eval/metrics.csv）から、Log Loss の図を作る。
# 使い方：python src/plot_main_results.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "data/processed/main_eval/metrics.csv")
C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1a1a19", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID})

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
for ax, (label, title, color) in zip(axes, [("決定点（主）", "Decision points (main)", C1), ("全イベント（副）", "All events (secondary)", C2)]):
    t = m[m["評価"].eq(label)].set_index("モデル").loc[["M0", "M1", "M2", "M3"]]
    y = range(len(t))[::-1]
    ax.errorbar(t["log_loss"], list(y), xerr=[t["log_loss"] - t["log_loss_lo"], t["log_loss_hi"] - t["log_loss"]],
                fmt="o", color=color, ecolor=color, elinewidth=2, capsize=0, markersize=7, markeredgecolor="white")
    for yi, v in zip(y, t["log_loss"]):
        ax.text(v, yi + 0.22, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), t.index); ax.set_ylim(-0.5, len(t) - 0.4)
    ax.set_title(f"{title}\nn={int(t['行'].iloc[0]):,} events, {int(t['学習者'].iloc[0]):,} learners", loc="left", fontsize=9, color=INK)
    ax.set_xlabel("log loss (lower is better), 95% learner bootstrap CI")
    ax.grid(axis="y", visible=False)
fig.tight_layout()
out = ROOT / "reports/figures/main_logloss.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out.relative_to(ROOT))
