# phase1_ednet_audit.ipynb のセクション0・1（セル0〜7）だけを実行し、出力をノートブックに保存する
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

nb_path = Path(sys.argv[1])
nb = nbformat.read(nb_path, as_version=4)
last = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("## 2."))

client = NotebookClient(nb, kernel_name="python3", timeout=600,
                        resources={"metadata": {"path": str(nb_path.parent)}})
with client.setup_kernel():
    for i in range(last):
        if nb.cells[i].cell_type == "code":
            client.execute_cell(nb.cells[i], i)
nbformat.write(nb, nb_path)
print(f"実行したセル: 0〜{last - 1}（セル{last}『## 2.』以降は未実行）")
