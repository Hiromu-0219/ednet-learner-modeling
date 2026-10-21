# 監査サンプル（KT3・KT4 各32人）の (action_type, item_type) 集計と、ノートブック セクション3 のトークン単位の遷移集計。
# 読み込み・並べ替えはノートブックのセル4・5と同じ（SEED=0、user_id→timestamp の安定ソート）。
# セクション3のセル13は CATEGORY_MAP で行を絞るが、カテゴリ定義は未決定なので全行を対象にする（ここだけノートブックと違う）。
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
DET_THRESHOLD = 0.95

def md(df, index=True, floatfmt="{:.3f}"):
    d = df.reset_index() if index else df
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)

def load(level):
    files = sorted((root / "data/audit" / level).glob("u*.csv"))
    rng = np.random.default_rng(0)
    audit_files = rng.choice(files, size=min(32, len(files)), replace=False)
    raw = pd.concat([pd.read_csv(f).assign(user_id=f.stem) for f in audit_files], ignore_index=True)
    raw = raw.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    raw["item_type"] = raw["item_id"].astype(str).str[0]
    return raw

out = {}
for level in ["KT3", "KT4"]:
    raw = load(level)
    print(f"\n#### {level}: 行数 {len(raw):,} / 学習者 {raw.user_id.nunique()} / カラム {list(raw.columns)}")

    pairs = raw.groupby(["action_type", "item_type"]).size().rename("件数").sort_values(ascending=False).to_frame()
    pairs["割合"] = pairs["件数"] / len(raw)
    print("\n[1] (action_type, item_type)\n" + md(pairs))

    ev = raw.copy()
    ev["token"] = ev["action_type"] + ":" + ev["item_type"]
    ev["next_token"] = ev.groupby("user_id")["token"].shift(-1)
    tr = ev.dropna(subset=["next_token"])
    P = pd.crosstab(tr["token"], tr["next_token"], normalize="index")
    stats = pd.DataFrame({
        "n": tr["token"].value_counts(),
        "max_p": P.max(axis=1),
        "argmax_next": P.idxmax(axis=1),
        "entropy_bits": -(P * np.log2(P.where(P > 0))).sum(axis=1),
    }).sort_values("n", ascending=False)
    stats["near_deterministic"] = stats["max_p"] >= DET_THRESHOLD
    share = stats.loc[stats.near_deterministic, "n"].sum() / stats["n"].sum()
    print(f"\n[2] 遷移数 {len(tr):,}。ほぼ確定 (max_p >= {DET_THRESHOLD}) な状態から出る遷移の割合: {share:.3f}")
    print(md(stats.rename_axis("token")))

    after = tr[tr["token"].str.startswith("submit")]
    dist = after["next_token"].value_counts().rename("件数").to_frame()
    dist["割合"] = dist["件数"] / len(after)
    print(f"\n[2'] submit の直後（{len(after):,} 件）\n" + md(dist.rename_axis("next_token")))

    # 補足：同時刻の行を含む遷移（KT4 で並び順が一定でないため）
    same_ts = (ev.groupby("user_id")["timestamp"].shift(-1) == ev["timestamp"]).loc[tr.index].mean()
    print(f"\n補足：次の行と同じ timestamp の遷移の割合 {same_ts:.3f}")
    out[level] = set(raw.user_id)

print("\n[3] user_id の重なり:", sorted(out["KT3"] & out["KT4"]) or "なし",
      f"（KT3 {len(out['KT3'])} 人 / KT4 {len(out['KT4'])} 人）")
m3 = json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))
print("KT3 manifest の audit_users と一致:", sorted(m3.get("audit_users", [])) == sorted(out["KT3"]))
