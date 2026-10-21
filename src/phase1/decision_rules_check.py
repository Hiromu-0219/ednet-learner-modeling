# 決定点の決め方（案）を KT4 の監査サンプルに当てて、残る件数と分類を確認する。
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
AUTO_SEC, GAP_MIN = 0.5, 60
NEXT_MAP = {"enter:e": "解説", "enter:b": "次の問題", "enter:l": "講義"}

def load(paths):
    raw = pd.concat([pd.read_csv(p).assign(user_id=p.stem) for p in paths], ignore_index=True)
    raw = raw.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    raw["token"] = raw["action_type"] + ":" + raw["item_id"].astype(str).str[0]
    g = raw.groupby("user_id")
    raw["next_token"], raw["next_ts"] = g["token"].shift(-1), g["timestamp"].shift(-1)
    gap = g["timestamp"].diff() / 60000
    raw["session_id"] = (gap.isna() | (gap > GAP_MIN)).groupby(raw["user_id"]).cumsum()
    raw["next_session"] = g["session_id"].shift(-1)
    raw["same_ts_next"] = raw["next_ts"] == raw["timestamp"]
    return raw

def decision_points(raw):
    dt = (raw["next_ts"] - raw["timestamp"]) / 1000
    auto = (raw.token == "submit:b") & (raw.next_token == "enter:e") & (dt <= AUTO_SEC)
    cand = raw[((raw.token == "submit:b") & ~auto) | (raw.token == "quit:e")].copy()
    cand["auto_excluded"] = 0
    stats = {"submit:b": int((raw.token == "submit:b").sum()), "自動表示で除外": int(auto.sum()),
             "quit:e": int((raw.token == "quit:e").sum())}
    diag = cand.source == "diagnosis"
    data_end = cand.next_token.isna()
    stats.update({"diagnosis で除外": int(diag.sum()), "データ末尾で除外": int((data_end & ~diag).sum())})
    dp = cand[~diag & ~data_end].copy()
    end = dp.next_session != dp.session_id
    dp["行動"] = np.where(end, "セッション終了", dp.next_token.map(NEXT_MAP).fillna("その他:" + dp.next_token))
    stats["決定点"] = len(dp)
    stats["同時刻の次の行を持つ決定点"] = int(dp.same_ts_next.sum())
    return dp, stats

for name, paths in [("KT4 監査32人（SEED=0）", sorted((root / "data/audit/KT4").glob("u*.csv"))),
                    ("KT4 同じ32人（KT3 監査ユーザー）", [root / "data/raw/KT4" / f"{u}.csv" for u in
                      json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]])]:
    raw = load(paths)
    dp, st = decision_points(raw)
    print(f"\n### {name}\n件数の流れ: {st}")
    print("決定点の種類 × 行動:\n", pd.crosstab(dp.token, dp["行動"], margins=True))
    per = dp.groupby("user_id").size().reindex(raw.user_id.unique(), fill_value=0)
    print("学習者ごとの決定点: 0件", int((per == 0).sum()), "人 / 1〜9件", int(((per > 0) & (per < 10)).sum()),
          "人 / 10件以上", int((per >= 10).sum()), "人 / 中央値", per.median(), "/ 最大", per.max())
    # submit→enter:e の時間差の谷（0.2〜0.5 秒）が KT4 でもあるか
    s = raw[(raw.token == "submit:b") & (raw.next_token == "enter:e")]
    d = (s.next_ts - s.timestamp) / 1000
    print("submit:b→enter:e 時間差: ≤0.2秒", round((d <= 0.2).mean(), 3), " 0.2〜0.5秒", round(((d > 0.2) & (d <= 0.5)).mean(), 3),
          " 0.5〜1秒", round(((d > 0.5) & (d <= 1)).mean(), 3), " >1秒", round((d > 1).mean(), 3), f"（{len(d)} 件）")
