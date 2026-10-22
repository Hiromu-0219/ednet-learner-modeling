# 適応曲線の探索的な確認（事前登録の外。reports/adaptation.md の「探索的な確認」）。
# 1. 未知の学習者の前半をすべて使ったときの M1・M2・M3（決定点）と、M3−M1 のブートストラップ区間
# 2. 前半の最初の行に diagnosis がどのくらい含まれるか
# 使い方：python src/adaptation_explore.py
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E
import adaptation as A
from pipeline import preprocess, read_logs, split_halves

ROOT = Path(__file__).resolve().parents[1]
users = json.loads((ROOT / "data/processed/main_sample/manifest.json").read_text(encoding="utf-8"))["users"]
data = split_halves(preprocess([ROOT / "data/raw/KT4" / f"{u}.csv" for u in users]))
dp_total = data[data.is_dp & data.target.notna()].groupby("user_id").size()
elig = np.array(sorted(dp_total[dp_total >= E.MIN_DP_PER_LEARNER].index))
unknown = set(np.random.default_rng(A.SPLIT_SEED).permutation(elig)[: len(elig) // 2])
m0, m1, kappa = A.population_fit(data[~data.user_id.isin(unknown)])
u = data[data.user_id.isin(unknown)]
first = u[u.half == "train"]
ok = first.groupby("user_id").size().pipe(lambda s: s[s >= max(A.K_GRID)].index)
first = first[first.user_id.isin(ok)]
test = u[u.half.eq("test") & u.user_id.isin(ok)].dropna(subset=["target"])
dp = test[test.is_dp]

st, order = E.learner_stats(A.fit_for_k(m0, m1, kappa, E.transition_counts(first)), dp)
for m in ["M1", "M2", "M3"]:
    print("前半すべて", m, round(float(E.metrics_from(st[m].sum(0))["log_loss"][0]), 4))
rng = np.random.default_rng(A.BOOT_SEED)
W = np.stack([np.bincount(rng.integers(0, len(order), len(order)), minlength=len(order)) for _ in range(E.N_BOOT)])
d = E.metrics_from(W @ st["M3"])["log_loss"] - E.metrics_from(W @ st["M1"])["log_loss"]
point = E.metrics_from(st["M3"].sum(0))["log_loss"][0] - E.metrics_from(st["M1"].sum(0))["log_loss"][0]
print("前半すべて M3−M1:", round(float(point), 4), np.round(np.quantile(d, [0.025, 0.975]), 4))
print("前半の行数の中央値:", int(first.groupby("user_id").size().median()))

raw = read_logs([ROOT / "data/raw/KT4" / f"{x}.csv" for x in sorted(ok)])
raw["token"] = raw.action_type + ":" + raw.item_id.astype(str).str[0]
key = ["user_id", "timestamp", "token"]
f2 = first.merge(raw[key + ["source"]].drop_duplicates(key), on=key, how="left")
f2["pos"] = f2.groupby("user_id").cumcount()
for lo, hi in [(0, 25), (25, 100), (100, 400), (400, 800)]:
    x = f2[(f2.pos >= lo) & (f2.pos < hi)]
    print(f"前半の {lo}〜{hi} 行目: diagnosis の割合 {x.source.eq('diagnosis').mean():.3f}")
lt = test.merge(raw[key + ["source"]].drop_duplicates(key), on=key, how="left")
print("後半: diagnosis の割合", round(lt.source.eq("diagnosis").mean(), 3))
