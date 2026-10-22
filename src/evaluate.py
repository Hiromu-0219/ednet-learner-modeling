"""M0〜M3 の推定・評価・生成（reports/evaluation_protocol.md の手順）。

使い方：
  python src/evaluate.py --users <ユーザー一覧の json> --data-dir data/raw/KT4 --out <出力フォルダ>
  python src/evaluate.py --audit --out <出力フォルダ>   # 監査サンプルでの動作確認（結果は報告しない）
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import AUTO_SEC, OUTCOMES, SESSION_GAP_MIN, STATES, TOKEN_OUTCOMES, TOKEN_TO_CATEGORY, TOKENS, preprocess, split_halves

MIN_DP_PER_LEARNER = 20                 # D7
ALPHA_GLOBAL = 0.5
ALPHA_M2 = 1 / len(TOKEN_OUTCOMES)       # 状態ごとの事前の重みの合計が1
KAPPA_GRID = np.logspace(-1, 4, 41)
N_BOOT, BOOT_SEED, SIM_SEED = 1000, 2, 3
K = len(OUTCOMES)                       # 評価するクラス（カテゴリ5つ）
KT = len(TOKEN_OUTCOMES)                # モデルが出す次の行動（14）
S_IDX = {s: i for i, s in enumerate(TOKENS)}          # 状態＝直前の行動（13）
T_IDX = {o: i for i, o in enumerate(TOKEN_OUTCOMES)}
O_IDX = {o: i for i, o in enumerate(OUTCOMES)}
AGG = np.zeros((KT, K))                 # 次の行動→カテゴリへの足し合わせ
for t, c in TOKEN_TO_CATEGORY.items():
    AGG[T_IDX[t], O_IDX[c]] = 1
END_T = T_IDX["end"]
_lgamma = np.vectorize(math.lgamma)


# ---------------------------------------------------------------- 推定
def transition_counts(t):
    """t: train の行 → 学習者ごとの 状態(13)×次の行動(14) の件数。"""
    t = t.dropna(subset=["target_token"])
    out = {}
    s = t["token"].map(S_IDX).to_numpy(); o = t["target_token"].map(T_IDX).to_numpy()
    for u, idx in t.groupby("user_id", sort=False).indices.items():
        m = np.zeros((len(TOKENS), KT)); np.add.at(m, (s[idx], o[idx]), 1); out[u] = m
    return out


def fit_kappa(counts_by_user, p):
    """状態ごとに、Dirichlet-多項分布の周辺尤度を最大にする κ を格子で選ぶ。"""
    kappas = []
    for si in range(len(TOKENS)):
        n = np.array([m[si] for m in counts_by_user.values()]); n = n[n.sum(1) > 0]
        N = n.sum(1)

        def loglik(kappa):
            a = kappa * p[si]
            return np.sum(math.lgamma(kappa) - _lgamma(kappa + N) + (_lgamma(n + a) - _lgamma(a)).sum(1))

        if len(n) == 0:
            kappas.append(np.nan); continue
        ll = np.array([loglik(k) for k in KAPPA_GRID])
        best = int(np.argmax(ll))
        fine = np.logspace(np.log10(KAPPA_GRID[max(best - 1, 0)]), np.log10(KAPPA_GRID[min(best + 1, len(KAPPA_GRID) - 1)]), 41)
        kappas.append(float(fine[np.argmax([loglik(k) for k in fine])]))
    return np.array(kappas)


def fit_models(train):
    counts = transition_counts(train)
    total = sum(counts.values())
    m0 = (total.sum(0) + ALPHA_GLOBAL); m0 = m0 / m0.sum()
    m1 = (total + ALPHA_GLOBAL); m1 = m1 / m1.sum(1, keepdims=True)
    kappa = fit_kappa(counts, m1)
    start = train.groupby(["user_id", "session_id"], sort=False)["token"].first().value_counts(normalize=True).reindex(TOKENS, fill_value=0)

    def predict(model, u):
        """状態(13)×次の行動(14) の確率。"""
        n = counts.get(u, np.zeros((len(TOKENS), KT)))
        if model == "M0": return np.tile(m0, (len(TOKENS), 1))
        if model == "M1": return m1
        if model == "M2": return (n + ALPHA_M2) / (n + ALPHA_M2).sum(1, keepdims=True)
        if model == "M3":
            kap = np.nan_to_num(kappa, nan=1.0)[:, None]          # 前半に遷移がない状態は κ=1
            return (n + kap * m1) / (n.sum(1, keepdims=True) + kap)
        raise ValueError(model)
    return {"m0": m0, "m1": m1, "kappa": kappa, "start": start, "counts": counts, "predict": predict}


# ---------------------------------------------------------------- 評価
MODELS = ["M0", "M1", "M2", "M3"]
N_BINS = 10


def learner_stats(fit, rows):
    """学習者ごとの十分統計量（足し合わせられる形）。返り値：model -> (学習者×統計量の配列, 学習者の一覧)。"""
    users = list(rows["user_id"].unique())
    s = rows["token"].map(S_IDX).to_numpy(); y = rows["target"].map(O_IDX).to_numpy()
    idx_by_user = rows.groupby("user_id", sort=False).indices
    out = {}
    for model in MODELS:
        st = np.zeros((len(users), 2 + 3 * N_BINS + 3 * K))
        for ui, u in enumerate(users):
            ix = idx_by_user[u]
            P = fit["predict"](model, u)[s[ix]] @ AGG             # 行×5（次の行動の確率をカテゴリに足し合わせる）
            yy = y[ix]
            pred = P.argmax(1); conf = P.max(1); correct = pred == yy
            st[ui, 0] = -np.log(P[np.arange(len(ix)), yy]).sum()
            st[ui, 1] = len(ix)
            b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
            st[ui, 2:2 + N_BINS] = np.bincount(b, minlength=N_BINS)
            st[ui, 2 + N_BINS:2 + 2 * N_BINS] = np.bincount(b, weights=conf, minlength=N_BINS)
            st[ui, 2 + 2 * N_BINS:2 + 3 * N_BINS] = np.bincount(b, weights=correct, minlength=N_BINS)
            o = 2 + 3 * N_BINS
            st[ui, o:o + K] = np.bincount(pred[correct], minlength=K)                 # TP
            st[ui, o + K:o + 2 * K] = np.bincount(pred, minlength=K)                  # 予測した数
            st[ui, o + 2 * K:o + 3 * K] = np.bincount(yy, minlength=K)                # 正解の数
        out[model] = st
    return out, users


def metrics_from(st):
    """st: 足し合わせた統計量（1次元）またはその行列（ブートストラップ×統計量）。"""
    st = np.atleast_2d(st)
    n = st[:, 1]
    bins_n = st[:, 2:2 + N_BINS]; bins_conf = st[:, 2 + N_BINS:2 + 2 * N_BINS]; bins_acc = st[:, 2 + 2 * N_BINS:2 + 3 * N_BINS]
    o = 2 + 3 * N_BINS
    tp, npred, ntrue = st[:, o:o + K], st[:, o + K:o + 2 * K], st[:, o + 2 * K:o + 3 * K]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0); rec = np.where(ntrue > 0, tp / ntrue, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = ntrue > 0
    return {"log_loss": st[:, 0] / n,
            "ece": np.abs(bins_conf - bins_acc).sum(1) / n,
            "accuracy": bins_acc.sum(1) / n,
            "macro_f1": (f1 * present).sum(1) / present.sum(1),
            "n": n}


def evaluate(fit, rows, label):
    stats, users = learner_stats(fit, rows)
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(N_BOOT)])
    point, boot = {}, {}
    for m in MODELS:
        point[m] = {k: float(v[0]) for k, v in metrics_from(stats[m].sum(0)).items()}
        boot[m] = metrics_from(W @ stats[m])
    table = []
    for m in MODELS:
        row = {"評価": label, "モデル": m, "学習者": len(users), "行": int(point[m]["n"])}
        for k in ["log_loss", "ece", "accuracy", "macro_f1"]:
            lo, hi = np.quantile(boot[m][k], [0.025, 0.975])
            row[k] = point[m][k]; row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        table.append(row)
    diffs = []
    for a, b in [("M3", "M1"), ("M3", "M2"), ("M2", "M1"), ("M1", "M0")]:
        for k in ["log_loss", "ece", "accuracy", "macro_f1"]:
            d = boot[a][k] - boot[b][k]
            lo, hi = np.quantile(d, [0.025, 0.975])
            diffs.append({"評価": label, "比較": f"{a}−{b}", "指標": k, "差": point[a][k] - point[b][k], "lo": lo, "hi": hi})
    return pd.DataFrame(table), pd.DataFrame(diffs)


# ---------------------------------------------------------------- 生成
TOK_CAT = np.array([O_IDX[TOKEN_TO_CATEGORY[t]] for t in TOKENS])


def simulate_sessions(P, start, n_sessions, rng, max_len=100_000):
    """行動（トークン）の単位で生成し、カテゴリの件数とセッション長を返す。"""
    cum_start = np.cumsum(start.to_numpy()); cum = np.cumsum(P, axis=1)
    lengths, counts = [], np.zeros(len(STATES))
    for _ in range(n_sessions):
        s = min(int(np.searchsorted(cum_start, rng.random(), side="right")), len(TOKENS) - 1)
        L = 1; counts[TOK_CAT[s]] += 1
        while L < max_len:
            o = min(int(np.searchsorted(cum[s], rng.random(), side="right")), KT - 1)
            if o == END_T: break
            s = o; L += 1; counts[TOK_CAT[s]] += 1
        lengths.append(L)
    return np.array(lengths), counts


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def reproduction(fit, data, eval_users):
    """後半にまるごと入るセッションについて、実データと M1・M3 の生成を比べる。"""
    test = data[data["half"].eq("test") & data["user_id"].isin(eval_users)]
    first_test = test.groupby(["user_id", "session_id"]).size()
    sess_in_train = data[data["half"].eq("train")].groupby(["user_id", "session_id"]).size()
    full = first_test[~first_test.index.isin(sess_in_train.index)]                 # 前半と分かれていないセッション
    rng = np.random.default_rng(SIM_SEED)
    rows, per_user = [], []
    real_rows = test.set_index(["user_id", "session_id"]).loc[full.index].reset_index()
    real_freq = real_rows["category"].value_counts(normalize=True).reindex(STATES, fill_value=0)
    rows.append({"系列": "実データ（後半）", "セッション": len(full), "平均": full.mean(), "中央値": full.median(),
                 "90%点": full.quantile(0.9), "終了率": len(full) / full.sum(), **real_freq.round(4).to_dict()})
    for model in ["M1", "M3"]:
        allL, allC = [], np.zeros(len(STATES))
        for u, n_s in full.groupby(level=0).size().items():
            L, C = simulate_sessions(fit["predict"](model, u), fit["start"], int(n_s), rng)
            allL.append(L); allC += C
            per_user.append({"user_id": u, "model": model, "sim_mean_len": L.mean(), "sim_end_rate": len(L) / L.sum()})
        L = np.concatenate(allL)
        rows.append({"系列": model, "セッション": len(L), "平均": L.mean(), "中央値": np.median(L), "90%点": np.quantile(L, 0.9),
                     "終了率": len(L) / L.sum(), **dict(zip(STATES, np.round(allC / allC.sum(), 4)))})
    real_u = full.groupby(level=0).agg(real_mean_len="mean", n="size", total="sum")
    real_u["real_end_rate"] = real_u["n"] / real_u["total"]
    pu = pd.DataFrame(per_user).pivot(index="user_id", columns="model")
    corr = []
    for model in ["M1", "M3"]:
        j = real_u.join(pu.xs(model, axis=1, level=1))
        corr.append({"モデル": model, "学習者": len(j),
                     "セッション長の平均 Spearman": spearman(j["real_mean_len"], j["sim_mean_len"]) if model == "M3" else np.nan,
                     "終了率 Spearman": spearman(j["real_end_rate"], j["sim_end_rate"]) if model == "M3" else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(corr)


# ---------------------------------------------------------------- 実行
def run(paths, out_dir, label, gap_min=SESSION_GAP_MIN, auto_sec=AUTO_SEC):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    d = preprocess(paths, gap_min=gap_min, auto_sec=auto_sec)
    data = split_halves(d)
    train = data[data["half"].eq("train")]
    fit = fit_models(train)
    test = data[data["half"].eq("test")].dropna(subset=["target"])
    dp_total = data[data["is_dp"] & data["target"].notna()].groupby("user_id").size()
    eligible = dp_total[dp_total >= MIN_DP_PER_LEARNER].index
    dp_test = test[test["is_dp"] & test["user_id"].isin(eligible)]
    ev_test = test[test["user_id"].isin(dp_test["user_id"].unique())]
    flow = {"セッション閾値(分)": gap_min, "D1 の秒数": auto_sec, "学習者（読み込み）": len(paths), "行（pay を除く）": len(d),
            "完了したセッションがある学習者": int(data["user_id"].nunique()),
            "完了したセッションの行": len(data), "前半の行": len(train),
            "決定点（完了したセッション）": int(dp_total.sum()),
            "D7 を満たす学習者": len(eligible), "後半に決定点がある評価対象の学習者": int(dp_test["user_id"].nunique()),
            "評価する決定点": len(dp_test), "評価する全イベント": len(ev_test),
            "κ（状態＝直前の行動）": {t: round(float(k), 3) for t, k in zip(TOKENS, fit["kappa"])}}
    tab_dp, diff_dp = evaluate(fit, dp_test, "決定点（主）")
    tab_ev, diff_ev = evaluate(fit, ev_test, "全イベント（副）")
    rep, corr = reproduction(fit, data, dp_test["user_id"].unique())
    pd.concat([tab_dp, tab_ev]).to_csv(out_dir / "metrics.csv", index=False)
    pd.concat([diff_dp, diff_ev]).to_csv(out_dir / "differences.csv", index=False)
    rep.to_csv(out_dir / "reproduction.csv", index=False)
    corr.to_csv(out_dir / "reproduction_by_learner.csv", index=False)
    dp_test["target"].value_counts().rename("件数").to_csv(out_dir / "dp_target_counts.csv")
    pd.DataFrame(fit["m1"], index=TOKENS, columns=TOKEN_OUTCOMES).to_csv(out_dir / "m1_token.csv")
    pd.DataFrame(fit["m1"] @ AGG, index=TOKENS, columns=OUTCOMES).to_csv(out_dir / "m1_category.csv")
    flow["実行時間(秒)"] = round(time.time() - t0, 1)
    (out_dir / "flow.json").write_text(json.dumps({"label": label, **flow}, ensure_ascii=False, indent=2), encoding="utf-8")
    return flow


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users"); ap.add_argument("--data-dir", default="data/raw/KT4")
    ap.add_argument("--audit", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--gap-min", type=float, default=SESSION_GAP_MIN)   # 感度分析（手順書8節）
    ap.add_argument("--auto-sec", type=float, default=AUTO_SEC)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.audit:
        u3 = json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]
        paths = sorted((root / "data/audit/KT4").glob("u*.csv")) + [root / "data/raw/KT4" / f"{u}.csv" for u in sorted(u3)]
        label = "監査サンプル（KT4 監査32人＋同じ32人）：動作確認用"
    else:
        users = json.loads(Path(a.users).read_text(encoding="utf-8"))["users"]
        paths = [Path(a.data_dir) / f"{u}.csv" for u in users]
        label = f"本番サンプル {len(users)}人"
    flow = run(paths, Path(a.out), label, gap_min=a.gap_min, auto_sec=a.auto_sec)
    print(json.dumps(flow, ensure_ascii=False, indent=2))
