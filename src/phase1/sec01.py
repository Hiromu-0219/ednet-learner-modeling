# 履歴用：最初の確認（2026-10-21）で使った。DATA_DIR は learner-simulation 内の旧パスのまま。
# phase1_ednet_audit.ipynb セクション0・1のコードをそのまま実行（DATA_DIR/KT_LEVEL のみ変更）
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

DATA_DIR = Path(r"C:\Users\hiro\Desktop\learner-simulation\data\raw\phase2_sample\KT3")
KT_LEVEL = "KT3"
N_AUDIT = 32
SEED = 0
SESSION_GAP_MIN = 60

rng = np.random.default_rng(SEED)
files = sorted(DATA_DIR.glob("u*.csv"))
print(f"学習者ファイル数: {len(files)}")
assert len(files) > 0

audit_files = rng.choice(files, size=min(N_AUDIT, len(files)), replace=False)
dfs = []
for f in audit_files:
    d = pd.read_csv(f)
    d.insert(0, "user_id", f.stem)
    dfs.append(d)
raw = pd.concat(dfs, ignore_index=True)
print("カラム:", list(raw.columns))
print(raw.dtypes)
print(f"\n行数: {len(raw):,} / 学習者数: {raw.user_id.nunique()}")
print("\n欠損率:\n", raw.isna().mean().round(3))
print(raw.head(10))

ts = raw["timestamp"]
print("\ntimestamp の桁数:", ts.astype(str).str.len().value_counts().to_dict())
raw["time"] = pd.to_datetime(ts, unit="ms")
print("期間:", raw["time"].min(), "〜", raw["time"].max())
raw = raw.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
dup = raw.duplicated(["user_id", "timestamp"], keep=False).mean()
print(f"同一タイムスタンプを持つ行の割合: {dup:.3f}")

raw["item_type"] = raw["item_id"].astype(str).str[0]
print("\naction_type:\n", raw["action_type"].value_counts())
print("\nitem_type:\n", raw["item_type"].value_counts())
if "source" in raw.columns:
    print("\nsource:\n", raw["source"].value_counts())
if "platform" in raw.columns:
    print("\nplatform:\n", raw["platform"].value_counts())
print("\n", pd.crosstab(raw["action_type"], raw["item_type"]))

# 補足（ノートブック外）: 欠損の内訳と、元ファイル順での時刻逆転
print("\nuser_answer 欠損率 by action_type:\n", raw.groupby("action_type")["user_answer"].apply(lambda s: s.isna().mean()).round(3))
inv = sum((pd.read_csv(f)["timestamp"].diff() < 0).sum() for f in files)
print("元ファイル順での時刻逆転の行数:", inv)
