# ノートブックの指定したセクション（「## N.」見出しの番号）のコードセルだけを、新しいカーネルで上から実行して保存する。
# 指定しなかったセクションのセルは実行せず、出力も変えない。
# 使い方：python src/phase1/run_sections.py phase1_ednet_audit_KT4.ipynb 0 1 4
import re
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

nb_path = Path(sys.argv[1])
targets = {int(s) for s in sys.argv[2:]}
nb = nbformat.read(nb_path, as_version=4)

section, run = None, []
for i, c in enumerate(nb.cells):
    m = re.match(r"## (\d+)\.", c.source) if c.cell_type == "markdown" else None
    if m:
        section = int(m.group(1))
    elif c.cell_type == "code" and section in targets:
        run.append(i)

client = NotebookClient(nb, kernel_name="python3", timeout=600, resources={"metadata": {"path": str(nb_path.parent)}})
with client.setup_kernel():
    for n, i in enumerate(run, 1):
        client.execute_cell(nb.cells[i], i, execution_count=n)
nbformat.write(nb, nb_path)
print(f"実行したセクション {sorted(targets)} のコードセル: {run}")
