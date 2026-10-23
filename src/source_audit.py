# source 列の値ごとの件数（監査サンプル、KT4）。自分で選んだ行動と推薦された行動を分けられるかの確認用。
# 使い方：python src/source_audit.py
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CATEGORY_MAP, read_logs

ROOT = Path(__file__).resolve().parents[1]
u3 = sorted(json.loads((ROOT / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"])
SETS = {"KT4 監査32人（SEED=0）": sorted((ROOT / "data/audit/KT4").glob("u*.csv")),
        "KT4 同じ32人": [ROOT / "data/raw/KT4" / f"{u}.csv" for u in u3]}
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

for name, paths in SETS.items():
    d = read_logs(paths)
    d["token"] = d["action_type"] + ":" + d["item_id"].astype(str).str[0]
    d["category"] = [CATEGORY_MAP.get(k) for k in zip(d["action_type"], d["item_id"].astype(str).str[0])]
    print(f"\n### {name}：{len(d):,}行・{d.user_id.nunique()}人")
    s = d.groupby("source", dropna=False).agg(行=("token", "size"), 学習者=("user_id", "nunique")).sort_values("行", ascending=False)
    s["割合"] = (s["行"] / len(d)).round(3)
    print(s.to_string())
    # 問題・解説・講義に入った行（enter）だけで、どの source から入ったか
    e = d[d["action_type"].eq("enter")]
    print("\nenter の行：source × 何に入ったか")
    print(pd.crosstab(e["source"].fillna("(欠損)"), e["token"], margins=True).to_string())
    # 学習者ごとに、source の種類を使った人数
    print("\n学習者ごと：その source の enter が1回以上ある人数")
    print(e.groupby("source")["user_id"].nunique().sort_values(ascending=False).to_string())
