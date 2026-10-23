# 本番サンプル（10,000人）で、source ごとの行数と、それを使った学習者の数だけを数える。行動の比較はしない。
# 使い方：python src/source_counts_main.py
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import preprocess, read_logs, split_halves

ROOT = Path(__file__).resolve().parents[1]
users = json.loads((ROOT / "data/processed/main_sample/manifest.json").read_text(encoding="utf-8"))["users"]
paths = [ROOT / "data/raw/KT4" / f"{u}.csv" for u in users]
d = read_logs(paths)
d["token"] = d["action_type"] + ":" + d["item_id"].astype(str).str[0]
d["source"] = d["source"].fillna("(欠損)")

# 評価で使った学習者の区分（完了したセッションがある / D7 を満たす）
data = split_halves(preprocess(paths))
dp_total = data[data["is_dp"] & data["target"].notna()].groupby("user_id").size()
completed = set(data["user_id"])
eligible = set(dp_total[dp_total >= 20].index)

s = d.groupby("source").agg(行=("token", "size"), 学習者=("user_id", "nunique")).sort_values("行", ascending=False)
s["行の割合"] = (s["行"] / len(d)).round(4)
s["完了したセッションがある学習者のうち"] = d[d["user_id"].isin(completed)].groupby("source")["user_id"].nunique()
s["評価対象（D7）の学習者のうち"] = d[d["user_id"].isin(eligible)].groupby("source")["user_id"].nunique()
s = s.fillna(0).astype({"完了したセッションがある学習者のうち": int, "評価対象（D7）の学習者のうち": int})

e = d[d["action_type"].eq("enter")]
enter = pd.crosstab(e["source"], e["token"])
per_user = e.groupby(["source", "user_id"]).size()
dist = per_user.groupby(level=0).apply(lambda x: pd.Series({"1件以上": (x >= 1).sum(), "5件以上": (x >= 5).sum(),
                                                              "20件以上": (x >= 20).sum(), "中央値": x.median()})).unstack()

out = ROOT / "data/processed/source_counts_main"; out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / "source_rows_learners.csv"); enter.to_csv(out / "source_enter.csv"); dist.to_csv(out / "source_enter_per_learner.csv")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
print(f"本番サンプル：{len(d):,}行・{d.user_id.nunique():,}人（完了したセッションがある {len(completed):,}人、D7 {len(eligible):,}人）\n")
print(s.to_string()); print("\nenter の行：source × 入った先\n" + enter.to_string())
print("\nenter が n 件以上ある学習者の数（source ごと）\n" + dist.to_string())
