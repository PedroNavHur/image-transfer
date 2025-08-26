# python/inspect_models.py
# Builds model-manifest for all ONNX models in public/models.
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]           # repo root
MODELS = ROOT / "public" / "models"                  # public/models

def inspect_one(path: Path):
    m = onnx.load(path.as_posix())
    g = m.graph
    inp = g.input[0]                                  # first input
    t = inp.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else None
            for d in t.shape.dim]                     # [N,*,*,*]
    layout = "unknown"
    H = W = None
    if len(dims) == 4:
        if dims[1] == 3:
            layout = "NCHW"
            H, W = dims[2], dims[3]
        elif dims[3] == 3:
            layout = "NHWC"
            H, W = dims[1], dims[2]
    return {
        "input": inp.name,
        "layout": layout,
        "H": H, "W": W,
        "dtype": str(t.elem_type),
        "ir_version": getattr(m, "ir_version", None),
        "opset": next((opset.version for opset in m.opset_import), None),
    }

out = {}
for p in MODELS.rglob("*.onnx"):
    rel = p.relative_to(MODELS).as_posix()            # e.g. "fns_candy.onnx"
    out[rel] = inspect_one(p)

print(json.dumps(out, indent=2))
