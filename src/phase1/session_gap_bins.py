# セクション4（セッション閾値）の補足：同じ学習者内の行動間隔を区間ごとに数え、閾値を変えたときのセッション数を出す。
# 計算はノートブックのセル18・19と同じ（timestamp の差、安定ソート）。監査サンプルだけを読む。
# 使い方：python src/phase1/session_gap_bins.py <プロジェクトのルート>
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
users3 = json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]
SETS = {"KT4 監査32人（SEED=0）": sorted((root / "data/audit/KT4").glob("u*.csv")),
        "KT4 同じ32人": [root / "data/raw/KT4" / f"{u}.csv" for u in sorted(users3)]}
EDGES = [10, 20, 30, 60, 90, 120, 240, 1440, np.inf]

for name, paths in SETS.items():
    raw = pd.concat([pd.read_csv(p).assign(user_id=p.stem) for p in paths], ignore_index=True)
    raw = raw.sort_values(["user_id", "timestamp"], kind="stable")
    gap = (raw.groupby("user_id")["timestamp"].diff() / 60000).dropna()
    bins = pd.cut(gap[gap > 10], EDGES, right=True).value_counts(sort=False)
    print(f"\n### {name}：間隔 {len(gap):,} 個のうち 10分超 {int((gap > 10).sum())} 個")
    print(" | ".join(f"{iv.left:g}〜{iv.right:g}分: {n}" for iv, n in bins.items()))
    rows = []
    for th in [10, 30, 60, 120, 240]:
        new = raw.groupby("user_id")["timestamp"].diff().div(60000).pipe(lambda g: g.isna() | (g > th))
        sizes = raw.groupby([raw["user_id"], new.groupby(raw["user_id"]).cumsum()]).size()
        rows.append({"閾値(分)": th, "セッション数": len(sizes), "セッションが1つだけの学習者": int((sizes.groupby(level=0).size() == 1).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
