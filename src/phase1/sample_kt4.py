# KT4 から監査サンプル32人を抽出する。抽出規則はノートブックのセル2・4と同じ：
#   files = sorted(DATA_DIR.glob("u*.csv")); rng = np.random.default_rng(SEED); rng.choice(files, size=32, replace=False)
import hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
src, dst = root / "data/raw/KT4", root / "data/audit/KT4"
SEED, N = 0, 32

files = sorted(src.glob("u*.csv"))
print("母集団ファイル数:", len(files))
rng = np.random.default_rng(SEED)
chosen = rng.choice(files, size=N, replace=False)

kt3_audit = {p.stem for p in (root / "data/audit").glob("u*.csv")}
overlap = sorted(p.stem for p in chosen if p.stem in kt3_audit)
print("KT3 監査32人との重複:", overlap or "なし")

dst.mkdir(parents=True, exist_ok=False)
entries = []
for p in chosen:
    shutil.copy2(p, dst / p.name)
    data = (dst / p.name).read_bytes()
    assert data == p.read_bytes()
    entries.append({"user_id": p.stem, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "rows": data.count(b"\n") - 1})

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "development audit only; exclude these users from future validation/test",
    "source_dir": "data/raw/KT4",
    "source_zip_sha256": "ae7b699f64980a2e795e3a7f02b144f32f822ba0a5b925826db98f2ef195fd40",
    "population_files": len(files),
    "selection": "files = sorted(Path('data/raw/KT4').glob('u*.csv')); np.random.default_rng(0).choice(files, size=32, replace=False)",
    "numpy_version": np.__version__,
    "seed": SEED,
    "overlap_with_kt3_audit": overlap,
    "audit_users": [e["user_id"] for e in entries],  # 抽出された順
    "files": entries,
}
(dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("抽出された順:", manifest["audit_users"])
print("合計行数:", sum(e["rows"] for e in entries), "/ 合計バイト:", sum(e["bytes"] for e in entries))
