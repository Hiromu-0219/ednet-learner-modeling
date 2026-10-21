# 決定点まわりの集計（監査サンプル）。読み込み・並べ替えはノートブックのセル4・5と同じ
# （user_id→timestamp の安定ソート。同一時刻の行は元ファイルの順を保つ）。
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
GAP_MIN = 60
KT3_DIR = root / "data/audit/KT3"
KT3_USERS = sorted(json.loads((KT3_DIR / "manifest.json").read_text(encoding="utf-8"))["audit_users"])

def md(df, index=True, floatfmt="{:.3f}"):
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(0.0 if abs(v) < 5e-4 else v)
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)

def load(paths):
    raw = pd.concat([pd.read_csv(p).assign(user_id=p.stem) for p in paths], ignore_index=True)
    raw = raw.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    raw["item_type"] = raw["item_id"].astype(str).str[0]
    raw["token"] = raw["action_type"] + ":" + raw["item_type"]
    g = raw.groupby("user_id")
    raw["next_token"] = g["token"].shift(-1)
    raw["next_ts"] = g["timestamp"].shift(-1)
    gap = g["timestamp"].diff() / 60000
    raw["session_id"] = (gap.isna() | (gap > GAP_MIN)).groupby(raw["user_id"]).cumsum()
    raw["next_session"] = g["session_id"].shift(-1)
    return raw

def token_stats(raw):
    tr = raw.dropna(subset=["next_token"])
    P = pd.crosstab(tr["token"], tr["next_token"], normalize="index")
    stats = pd.DataFrame({"n": tr["token"].value_counts(), "max_p": P.max(axis=1), "argmax_next": P.idxmax(axis=1),
                          "entropy_bits": -(P * np.log2(P.where(P > 0))).sum(axis=1)}).sort_values("n", ascending=False)
    return tr, stats

kt3 = load([KT3_DIR / f"{u}.csv" for u in KT3_USERS])
kt4s = load([root / "data/raw/KT4" / f"{u}.csv" for u in KT3_USERS])
kt4r = load(sorted((root / "data/audit/KT4").glob("u*.csv")))

# ---------- 1. 解説の自動表示 ----------
print("## 1. submit:b → enter:e（KT3 監査32人）")
sub = kt3[kt3.token == "submit:b"].copy()
sub["next_is_e"] = sub["next_token"] == "enter:e"
d = sub[sub.next_is_e]
dt = (d["next_ts"] - d["timestamp"]) / 1000
print(f"submit:b {len(sub):,} 件のうち、直後が enter:e {len(d):,} 件。時間差（秒）: 中央値 {dt.median():.3f}, "
      f"平均 {dt.mean():.3f}, 最小 {dt.min():.3f}, 最大 {dt.max():.1f}, 0〜1秒の割合 {(dt <= 1).mean():.3f}, 0秒ちょうど {(dt == 0).mean():.3f}")
print("分位点:", {q: round(dt.quantile(q), 3) for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]})
bins = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 60, np.inf]
labels = ["0〜0.05", "0.05〜0.1", "0.1〜0.2", "0.2〜0.5", "0.5〜1", "1〜2", "2〜5", "5〜10", "10〜60", "60〜"]
h = pd.cut(dt, bins=bins, labels=labels, include_lowest=True).value_counts(sort=False).rename("件数").to_frame()
h["割合"] = h["件数"] / len(dt)
h["棒"] = (h["割合"] * 50).round().astype(int).map(lambda k: "█" * k)
print(md(h.rename_axis("時間差（秒）")))
# 比較：submit:b → enter:b
d2 = sub[sub.next_token == "enter:b"]
dt2 = (d2["next_ts"] - d2["timestamp"]) / 1000
print(f"比較：直後が enter:b の {len(d2):,} 件の時間差 中央値 {dt2.median():.3f} 秒、0〜1秒の割合 {(dt2 <= 1).mean():.3f}")
for col in ["source", "platform"]:
    t = sub.groupby(col).agg(submit件数=("next_is_e", "size"), enter_e件数=("next_is_e", "sum"))
    t["enter_e率"] = t["enter_e件数"] / t["submit件数"]
    print(f"\n{col} 別（submit:b の行の {col}）\n" + md(t.sort_values("submit件数", ascending=False)))
# 学習者ごとの enter:e 率（自動表示なら全員ほぼ同じ値になるはず）
pu = sub.groupby("user_id")["next_is_e"].agg(["size", "mean"])
print(f"\n学習者ごとの submit 後 enter:e 率（submit のある {len(pu)} 人）: 最小 {pu['mean'].min():.3f} / 中央値 {pu['mean'].median():.3f} / 最大 {pu['mean'].max():.3f}")
print("enter:e 率が 1.0 の人:", int((pu['mean'] == 1).sum()), " 0.0 の人:", int((pu['mean'] == 0).sum()))

# ---------- 2. 決定点 ----------
def classify(raw, token, mapping):
    x = raw[raw.token == token].copy()
    end = x["next_token"].isna() | (x["next_session"] != x["session_id"])
    x["行動"] = np.where(end, "セッション終了", x["next_token"].map(mapping).fillna("その他:" + x["next_token"].fillna("")))
    x["データ末尾"] = x["next_token"].isna()
    return x

def summarize(x, order, title):
    print(f"\n### {title}（{len(x):,} 件、学習者 {x.user_id.nunique()} 人）")
    tot = x["行動"].value_counts().reindex(order + sorted(set(x["行動"]) - set(order)), fill_value=0)
    per = pd.crosstab(x["user_id"], x["行動"], normalize="index").reindex(columns=tot.index, fill_value=0)
    t = pd.DataFrame({"件数": tot, "全体の割合": tot / len(x), "学習者別_平均": per.mean(), "学習者別_中央値": per.median(),
                      "学習者別_SD": per.std(ddof=1), "学習者別_分散": per.var(ddof=1), "学習者別_最小": per.min(), "学習者別_最大": per.max()})
    print(md(t.rename_axis("行動")))
    print(f"うちデータ末尾（その学習者の最後の行）によるセッション終了: {int(x['データ末尾'].sum())} 件")
    return per

print("\n## 2. 決定点（KT3 監査32人、セッション閾値60分）")
s = classify(kt3, "submit:b", {"enter:e": "解説", "enter:b": "次の問題", "enter:l": "講義"})
summarize(s, ["解説", "次の問題", "講義", "セッション終了"], "submit:b の後")
q = classify(kt3, "quit:e", {"enter:b": "次の問題", "enter:l": "講義"})
summarize(q, ["次の問題", "講義", "セッション終了"], "quit:e の後")
cnt = pd.DataFrame({"submit:b": s.groupby("user_id").size(), "quit:e": q.groupby("user_id").size()}).reindex(KT3_USERS).fillna(0).astype(int)
cnt["合計"] = cnt.sum(axis=1)
print("\n### 学習者ごとの決定点の件数（32人）")
print(md(cnt.describe().T[["min", "25%", "50%", "75%", "max", "mean"]].rename_axis("決定点"), floatfmt="{:.1f}"))
print("合計の件数ごとの人数:", pd.cut(cnt["合計"], [-1, 0, 10, 50, 100, 200, 500, np.inf],
      labels=["0", "1〜10", "11〜50", "51〜100", "101〜200", "201〜500", "501〜"]).value_counts(sort=False).to_dict())
print("セッション数（60分）: 学習者あたり", kt3.groupby("user_id")["session_id"].max().describe()[["min", "50%", "max"]].to_dict())

# ---------- 3. 同じ32人の KT4 ----------
print("\n## 3. 同じ32人の KT4（data/raw/KT4）")
print(f"行数 {len(kt4s):,} / 学習者 {kt4s.user_id.nunique()} / 次の行と同じ timestamp の遷移の割合 "
      f"{(kt4s['next_ts'] == kt4s['timestamp']).loc[kt4s.next_token.notna()].mean():.3f}")
pairs = kt4s.groupby(["action_type", "item_type"]).size().rename("件数").sort_values(ascending=False).to_frame()
pairs["割合"] = pairs["件数"] / len(kt4s)
print(md(pairs))
tr, st4 = token_stats(kt4s)
print(f"\n遷移 {len(tr):,}\n" + md(st4.rename_axis("token")))
a = tr[tr.token.str.startswith("submit")]["next_token"].value_counts()
print(f"\nsubmit 直後（{a.sum():,} 件）:", {k: f"{v} ({v / a.sum():.3f})" for k, v in a.items()})

# ---------- 4. 閾値 ----------
print("\n## 4. 「ほぼ確定」の割合と閾値")
rows = {}
for name, raw in [("KT3 監査32人", kt3), ("KT4 同じ32人", kt4s), ("KT4 監査32人（SEED=0）", kt4r)]:
    _, st = token_stats(raw)
    rows[name] = {f"≥{th:.2f}": st.loc[st.max_p >= th, "n"].sum() / st["n"].sum() for th in [0.90, 0.95, 0.99]}
    rows[name]["該当トークン(≥0.90)"] = ", ".join(st.index[st.max_p >= 0.90])
print(md(pd.DataFrame(rows).T.rename_axis("データ")))
