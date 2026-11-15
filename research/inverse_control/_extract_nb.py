"""One-shot: dump Pulp Fiction notebook cells we need. Not part of the pipeline."""
import json
from pathlib import Path

nb = json.loads(Path(r"C:\Users\argsw\Downloads\gb-vpp-pulp-fiction.ipynb").read_text(encoding="utf-8"))
out = Path(__file__).resolve().parent / "_nb_cells"
out.mkdir(exist_ok=True)
print("n_cells", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first = src.strip().splitlines()[0][:100] if src.strip() else "(empty)"
    print(f"{i:02d} {c['cell_type']:8s} {len(src):5d}  {first}")
    (out / f"cell_{i:02d}_{c['cell_type']}.txt").write_text(src, encoding="utf-8")
