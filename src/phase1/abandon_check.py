# 「あきらめる」にあたる行動がログに現れるかの確認（監査サンプル、KT4）。
# 問題の試行 = enter:b から、次の enter（b/e/l）・セッション終了・データ末尾までの区間。
# その区間に submit:b（同じ bundle）と respond:q があるかを数える。セッション区切りは60分（D4 と同じ）。
# 使い方：python src/phase1/abandon_check.py <プロジェクトのルート>
import json, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
GAP_MIN = 60
users3 = json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]
SETS = {"KT4 監査32人（SEED=0）": sorted((root / "data/audit/KT4").glob("u*.csv")),
        "KT4 同じ32人": [root / "data/raw/KT4" / f"{u}.csv" for u in sorted(users3)]}

for name, paths in SETS.items():
    raw = pd.concat([pd.read_csv(p).assign(user_id=p.stem) for p in paths], ignore_index=True)
    raw = raw.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    raw["token"] = raw["action_type"] + ":" + raw["item_id"].astype(str).str[0]
    gap = raw.groupby("user_id")["timestamp"].diff() / 60000
    raw["session_id"] = (gap.isna() | (gap > GAP_MIN)).groupby(raw["user_id"]).cumsum()

    attempts = []
    for (u, s), d in raw.groupby(["user_id", "session_id"], sort=False):
        cur = None
        for row in d.itertuples(index=False):
            if row.action_type == "enter":
                if cur: cur["end"] = "次の enter"; attempts.append(cur); cur = None
                if row.token == "enter:b":
                    cur = {"user_id": u, "bundle": row.item_id, "source": row.source, "submit": False, "respond": 0, "end": None}
            elif cur is not None:
                if row.token == "submit:b" and row.item_id == cur["bundle"]:
                    cur["submit"] = True; cur["end"] = "submit"; attempts.append(cur); cur = None
                elif row.token == "respond:q":
                    cur["respond"] += 1
        if cur:
            cur["end"] = "セッション終了"; attempts.append(cur)
    a = pd.DataFrame(attempts)
    no_submit = a[~a["submit"]]
    print(f"\n### {name}：問題の試行 {len(a):,} 件（学習者 {a.user_id.nunique()} 人）")
    print(f"- 提出せずに終わった試行: {len(no_submit)} 件。終わり方の内訳: {no_submit['end'].value_counts().to_dict()}")
    print(f"- 提出したが respond:q が0件の試行: {int((a['submit'] & a['respond'].eq(0)).sum())} 件")
    print(f"- 試行あたりの respond:q の件数: {a['respond'].value_counts().sort_index().to_dict()}")
    print(f"- 対象外（pay）の行: {int(raw['token'].eq('pay:p').sum())} 行")
