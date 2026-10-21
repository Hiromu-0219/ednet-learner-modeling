# 本番サンプルの抽出（reports/evaluation_protocol.md の2節）。
# KT4 全体から監査ユーザー64人を除き、default_rng(1) で10,000人を抽出して manifest に記録する。
# 使い方：python src/sample_main.py
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data/raw/KT4", ROOT / "data/processed/main_sample"
SEED, N = 1, 10_000

audit = set(json.loads((ROOT / "data/audit/KT3/manifest.json").read_text(encoding="utf-8"))["audit_users"]) | \
        set(json.loads((ROOT / "data/audit/KT4/manifest.json").read_text(encoding="utf-8"))["audit_users"])
files = sorted(SRC.glob("u*.csv"))
pool = [f for f in files if f.stem not in audit]
chosen = np.random.default_rng(SEED).choice(pool, size=N, replace=False)
users = [f.stem for f in chosen]
digest = hashlib.sha256("".join(f"{f.stem}:{hashlib.sha256(f.read_bytes()).hexdigest()}\n" for f in sorted(chosen)).encode()).hexdigest()

OUT.mkdir(parents=True, exist_ok=True)
manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_dir": "data/raw/KT4",
            "source_zip_sha256": "ae7b699f64980a2e795e3a7f02b144f32f822ba0a5b925826db98f2ef195fd40",
            "population_files": len(files), "excluded_audit_users": len(audit & {f.stem for f in files}),
            "pool": len(pool), "seed": SEED, "n": N, "numpy_version": np.__version__,
            "selection": "pool = sorted(data/raw/KT4/u*.csv) minus audit users; np.random.default_rng(1).choice(pool, size=10000, replace=False)",
            "files_sha256_digest": digest, "users": users}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: v for k, v in manifest.items() if k != "users"}, "先頭5人:", users[:5])
