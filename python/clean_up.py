#!/usr/bin/env python3
"""
Clean all Fast Neural Style ONNX models.

Default behavior:
  - Processes ../public/models/fns_*.onnx
  - Writes side-by-side files with `.fixed.onnx` suffix.

Usage:
  python clean_up.py
  python clean_up.py --inplace
  python clean_up.py --glob "public/models/fns_*.onnx"
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import onnx
from onnx import checker, shape_inference


def clean_model(path: Path) -> onnx.ModelProto:
    m = onnx.load(str(path))

    # Remove real initializers from graph inputs (exporters sometimes include them)
    init_names = {init.name for init in m.graph.initializer}
    filtered_inputs = [vi for vi in m.graph.input if vi.name not in init_names]
    if len(filtered_inputs) != len(m.graph.input):
        m.graph.ClearField("input")
        m.graph.input.extend(filtered_inputs)

    # ONNX Runtime optimizers assume IR >= 4 for "initializers not in inputs"
    if m.ir_version < 4:
        m.ir_version = 4

    # Optional: shape inference (non-fatal)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print(f"[warn] shape inference skipped for {path.name}: {e}")

    checker.check_model(m)
    return m


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_pattern = repo_root / "public" / "models" / "fns_*.onnx"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default=str(default_pattern),
        help="Glob of ONNX files to process (default: public/models/fns_*.onnx)",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file instead of writing *.fixed.onnx",
    )
    ap.add_argument(
        "--suffix",
        default=".fixed",
        help="Suffix inserted before .onnx when not --inplace (default: .fixed)",
    )
    args = ap.parse_args()

    files = [Path(p) for p in glob(args.glob)]
    if not files:
        print(f"[info] no files matched: {args.glob}")
        return

    print(f"[info] processing {len(files)} file(s)")
    for inp in files:
        try:
            model = clean_model(inp)
            if args.inplace:
                outp = inp
            else:
                outp = inp.with_name(inp.stem + args.suffix + inp.suffix)
            onnx.save(model, str(outp))
            print(f"[ok] {inp.name} -> {outp.name}")
        except Exception as e:
            print(f"[err] {inp.name}: {e}")

    print("[done]")


if __name__ == "__main__":
    main()
