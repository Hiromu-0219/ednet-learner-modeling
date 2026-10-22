# 実行済みノートブックの図を PNG として書き出す（スライド用）。
# 使い方：python src/export_notebook_figures.py
import base64
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures"
# (ノートブック, 何番目の図か, 書き出す名前)
TARGETS = [("notebooks/phase1_decisions_evidence.ipynb", 1, "evidence_d1_gap_after_submit.png"),
           ("notebooks/phase1_decisions_evidence.ipynb", 2, "evidence_d2_enter_e_rate_by_source.png"),
           ("notebooks/phase1_decisions_evidence.ipynb", 3, "evidence_d5_near_deterministic.png"),
           ("notebooks/phase1_decisions_evidence.ipynb", 4, "evidence_d7_decision_points_per_learner.png")]

OUT.mkdir(parents=True, exist_ok=True)
for nb_path, nth, name in TARGETS:
    nb = nbformat.read(ROOT / nb_path, as_version=4)
    images = [o.data["image/png"] for c in nb.cells if c.cell_type == "code" for o in c.outputs if "image/png" in o.get("data", {})]
    (OUT / name).write_bytes(base64.b64decode(images[nth - 1]))
    print("wrote", (OUT / name).relative_to(ROOT))
