"""副評価：未知学習者の適応曲線（reports/evaluation_protocol.md の9節）。

使い方：
  python src/adaptation.py --users data/processed/main_sample/manifest.json --out data/processed/adaptation
  python src/adaptation.py --audit --out data/processed/adaptation_audit   # 動作確認（結果は報告しない）
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E
from pipeline import preprocess, split_halves

K_GRID = [0, 25, 50, 100, 200, 400, 800]
SPLIT_SEED, BOOT_SEED = 4, 5


def population_fit(rows):
    """未知の学習者を除いた行で、M0・M1・κ を推定する（evaluate.fit_models と同じ方法）。"""
    f = E.fit_models(rows)
    return f["m0"], f["m1"], f["kappa"]


def fit_for_k(m0, m1, kappa, counts):
    kap = np.nan_to_num(kappa, nan=1.0)[:, None]
    zero = np.zeros_like(m1)

    def predict(model, u):
        n = counts.get(u, zero)
        if model == "M0": return np.tile(m0, (m1.shape[0], 1))
        if model == "M1": return m1
        if model == "M2": return (n + E.ALPHA_M2) / (n + E.ALPHA_M2).sum(1, keepdims=True)
        if model == "M3": return (n + kap * m1) / (n.sum(1, keepdims=True) + kap)
        raise ValueError(model)
    return {"predict": predict}


def curve(m0, m1, kappa, first_half, eval_rows, label):
    users = list(eval_rows["user_id"].unique())
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(E.N_BOOT)])
    out = []
    for k in K_GRID:
        adapt = first_half.groupby("user_id", sort=False).head(k)
        fit = fit_for_k(m0, m1, kappa, E.transition_counts(adapt))
        stats, order = E.learner_stats(fit, eval_rows)
        assert order == users
        point = {m: E.metrics_from(stats[m].sum(0)) for m in E.MODELS}
        boot = {m: E.metrics_from(W @ stats[m]) for m in E.MODELS}
        row = {"評価": label, "k": k, "学習者": len(users), "行": int(point["M1"]["n"][0])}
        for m in E.MODELS:
            for met in ["log_loss", "ece", "accuracy"]:
                row[f"{m}_{met}"] = float(point[m][met][0])
        for a, b in [("M3", "M1"), ("M3", "M2"), ("M2", "M1")]:
            for met in ["log_loss", "ece", "accuracy"]:
                d = boot[a][met] - boot[b][met]
                row[f"{a}-{b}_{met}"] = float(point[a][met][0] - point[b][met][0])
                row[f"{a}-{b}_{met}_lo"], row[f"{a}-{b}_{met}_hi"] = np.quantile(d, [0.025, 0.975])
        out.append(row)
    return pd.DataFrame(out)


def run(paths, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = split_halves(preprocess(paths))
    dp_total = data[data["is_dp"] & data["target"].notna()].groupby("user_id").size()
    eligible = np.array(sorted(dp_total[dp_total >= E.MIN_DP_PER_LEARNER].index))
    perm = np.random.default_rng(SPLIT_SEED).permutation(eligible)
    unknown = set(perm[: len(perm) // 2])
    m0, m1, kappa = population_fit(data[~data["user_id"].isin(unknown)])
    u = data[data["user_id"].isin(unknown)]
    first = u[u["half"].eq("train")]
    long_enough = first.groupby("user_id").size().pipe(lambda s: s[s >= max(K_GRID)].index)
    first = first[first["user_id"].isin(long_enough)]
    test = u[u["half"].eq("test") & u["user_id"].isin(long_enough)].dropna(subset=["target"])
    dp_test = test[test["is_dp"]]
    test = test[test["user_id"].isin(dp_test["user_id"].unique())]
    res = pd.concat([curve(m0, m1, kappa, first, dp_test, "決定点（主）"), curve(m0, m1, kappa, first, test, "全イベント（副）")])
    res.to_csv(out_dir / "adaptation.csv", index=False)
    flow = {"label": label, "D7 を満たす学習者": len(eligible), "未知の学習者": len(unknown),
            f"前半が{max(K_GRID)}行以上の未知の学習者（評価対象）": int(dp_test["user_id"].nunique()),
            "評価する決定点": len(dp_test), "評価する全イベント": len(test),
            "全体モデルの推定に使った行": int((~data["user_id"].isin(unknown)).sum()),
            "κ": {t: round(float(k), 3) for t, k in zip(E.TOKENS, kappa)}}
    (out_dir / "flow.json").write_text(json.dumps(flow, ensure_ascii=False, indent=2), encoding="utf-8")
    return flow, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users"); ap.add_argument("--audit", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.audit:
        u3 = json.loads((root / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]
        paths = sorted((root / "data/audit/KT4").glob("u*.csv")) + [root / "data/raw/KT4" / f"{u}.csv" for u in sorted(u3)]
        label = "監査サンプル：動作確認用"
    else:
        users = json.loads(Path(a.users).read_text(encoding="utf-8"))["users"]
        paths = [root / "data/raw/KT4" / f"{u}.csv" for u in users]
        label = f"本番サンプル {len(users)}人"
    flow, res = run(paths, Path(a.out), label)
    print(json.dumps(flow, ensure_ascii=False, indent=2))
