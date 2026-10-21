"""前処理パイプライン（reports/evaluation_protocol.md の固定した定義）。

preprocess(paths) は学習者ごとの CSV を読み、1行=1行動の DataFrame を返す。
  category        : solve / review / lecture / aux（pay の行は落とす）
  session_id      : 60分区切りのセッション番号（学習者ごと）
  censored_session: データ末尾で打ち切られたセッション（各学習者の最後のセッション）なら True（D8）
  target          : 次のカテゴリ。セッションの最後の行は "end"。各学習者の最後の行は NaN
  target_token    : 次の行動（トークン）。end・NaN の扱いは target と同じ
  is_dp           : 決定点（D1〜D3。D4 のデータ末尾は target が NaN なので、使う側で除く）
"""
from pathlib import Path
import numpy as np
import pandas as pd

SESSION_GAP_MIN = 60
AUTO_SEC = 0.5                     # D1
DIAGNOSIS_SOURCE = "diagnosis"     # D2
STATES = ["solve", "review", "lecture", "aux"]
OUTCOMES = STATES + ["end"]
CATEGORY_MAP = {
    ("enter", "b"): "solve", ("respond", "q"): "solve", ("submit", "b"): "solve",
    ("enter", "e"): "review", ("quit", "e"): "review",
    ("enter", "l"): "lecture", ("quit", "l"): "lecture", ("play_video", "l"): "lecture", ("pause_video", "l"): "lecture",
    ("play_audio", "b"): "aux", ("pause_audio", "b"): "aux", ("erase_choice", "q"): "aux", ("undo_erase_choice", "q"): "aux",
    ("pay", "p"): None,
}
TOKENS = [f"{a}:{i}" for (a, i), c in CATEGORY_MAP.items() if c is not None]   # モデルの状態（13種類）
TOKEN_OUTCOMES = TOKENS + ["end"]
TOKEN_TO_CATEGORY = {f"{a}:{i}": c for (a, i), c in CATEGORY_MAP.items() if c is not None} | {"end": "end"}


def read_logs(paths):
    d = pd.concat([pd.read_csv(p, usecols=["timestamp", "action_type", "item_id", "source"]).assign(user_id=Path(p).stem)
                   for p in paths], ignore_index=True)
    return d.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)   # D6


def preprocess(paths, gap_min=SESSION_GAP_MIN, auto_sec=AUTO_SEC):
    d = read_logs(paths)
    item_type = d["item_id"].astype(str).str[0]
    d["token"] = d["action_type"] + ":" + item_type

    # 決定点（生の行の並びで判定する）
    g = d.groupby("user_id", sort=False)
    next_token = g["token"].shift(-1)
    dt_next = (g["timestamp"].shift(-1) - d["timestamp"]) / 1000
    is_submit, is_quit = d["token"].eq("submit:b"), d["token"].eq("quit:e")
    auto = is_submit & next_token.eq("enter:e") & (dt_next <= auto_sec)            # D1
    d["is_dp"] = ((is_submit & ~auto) | is_quit) & d["source"].ne(DIAGNOSIS_SOURCE)   # D2

    # カテゴリ（pay を落とす）
    cats = [CATEGORY_MAP.get(k, "UNMAPPED") for k in zip(d["action_type"], item_type)]
    d["category"] = cats
    unmapped = d.loc[d["category"].eq("UNMAPPED"), "token"].unique()
    if len(unmapped):
        raise ValueError(f"未割り当ての組: {list(unmapped)}")
    d = d[d["category"].notna()].reset_index(drop=True)

    # セッション・予測対象
    g = d.groupby("user_id", sort=False)
    gap = g["timestamp"].diff() / 60000
    d["session_id"] = (gap.isna() | (gap > gap_min)).groupby(d["user_id"]).cumsum()
    next_cat = g["category"].shift(-1)
    next_sid = d.groupby("user_id", sort=False)["session_id"].shift(-1)
    d["target"] = next_cat.where(next_sid.eq(d["session_id"]), "end")
    d.loc[next_cat.isna(), "target"] = np.nan                                       # データ末尾（D4）
    next_tok = g["token"].shift(-1)
    d["target_token"] = next_tok.where(next_sid.eq(d["session_id"]), "end")
    d.loc[next_tok.isna(), "target_token"] = np.nan
    d["censored_session"] = d["session_id"].eq(d.groupby("user_id", sort=False)["session_id"].transform("max"))  # D8
    return d[["user_id", "timestamp", "token", "category", "session_id", "censored_session", "target", "target_token", "is_dp"]]


def split_halves(d):
    """D8 で完了したセッションだけを残し、学習者ごとに前半 floor(n/2) 行を train、残りを test にする。"""
    c = d[~d["censored_session"]].copy()
    pos = c.groupby("user_id", sort=False).cumcount()
    n = c.groupby("user_id", sort=False)["category"].transform("size")
    c["half"] = np.where(pos < n // 2, "train", "test")
    return c
