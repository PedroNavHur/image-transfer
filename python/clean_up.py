# clean_up.py
# Remove initializers from graph inputs in an ONNX model.
# Usage: python clean_up.py
# Requires: onnx
# Example: python clean_up.py
# This script loads an ONNX model from 'in.onnx', removes any initializers from the graph inputs,
# ensures the IR version is at least 4, optionally re-infers shapes, checks the model, and saves the cleaned model to 'out.fixed.onnx'.
#
# Used for cleaning up the Fast Neural Style Transfer ONNX models.
import onnx
from onnx import checker, shape_inference

inp = "in.onnx"
out = "out.fixed.onnx"

m = onnx.load(inp)

# names of true initializers (weights)
init_names = {init.name for init in m.graph.initializer}

# keep only real inputs (exclude any that are actually initializers)
original_inputs = list(m.graph.input)
filtered_inputs = [vi for vi in original_inputs if vi.name not in init_names]

# replace the graph.input list safely
m.graph.ClearField("input")
m.graph.input.extend(filtered_inputs)

# ensure IR >= 4 (where initializers don't have to appear in inputs)
if m.ir_version < 4:
  m.ir_version = 4

# (optional) try to re-infer shapes; OK to skip if it fails
try:
  m = shape_inference.infer_shapes(m)
except Exception as e:
  print("Shape inference skipped:", e)

checker.check_model(m)
onnx.save(m, out)
print(f"Wrote {out}")
