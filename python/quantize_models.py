#!/usr/bin/env python3
import argparse
import glob
import os
import random
from typing import Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from PIL import Image

# Pillow resample enum compat
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except Exception:
    RESAMPLE_BICUBIC = Image.BICUBIC

# ---------- Input inspection & preprocessing ----------

def _extract_hw_from_dims(dims, layout: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Given dims (length 4) and layout, return (H, W) if both are fixed (>0), else (None, None).
    NHWC: [N, H, W, C]
    NCHW: [N, C, H, W]
    """
    if len(dims) != 4:
        return (None, None)
    if layout == "NHWC":
        h, w = dims[1], dims[2]
    else:  # NCHW
        h, w = dims[2], dims[3]
    if (h or 0) > 0 and (w or 0) > 0:
        return (h, w)
    return (None, None)

def get_input_info(model_path: str):
    """
    Returns:
      m: onnx.ModelProto
      input_name: str
      layout: 'NHWC' or 'NCHW'
      dtype: 'float32' or 'uint8'
      req_h: Optional[int]  # required input height if fixed in ONNX (else None)
      req_w: Optional[int]  # required input width if fixed in ONNX (else None)
    """
    m = onnx.load(model_path)
    if not m.graph.input:
        raise ValueError(f"No inputs found in model: {model_path}")
    inp = m.graph.input[0]
    tinfo = inp.type.tensor_type
    dims = [d.dim_value for d in tinfo.shape.dim] if tinfo.shape.dim else []
    elem_type = tinfo.elem_type

    # Layout heuristic
    layout = "NCHW"
    if len(dims) == 4:
        if dims[-1] == 3:
            layout = "NHWC"
        elif dims[1] == 3:
            layout = "NCHW"
        else:
            layout = "NHWC" if ":0" in inp.name.lower() else "NCHW"

    # Dtype
    if elem_type == TensorProto.FLOAT:
        dtype = "float32"
    elif elem_type == TensorProto.UINT8:
        dtype = "uint8"
    else:
        dtype = "float32"

    req_h, req_w = _extract_hw_from_dims(dims, layout)
    return m, inp.name, layout, dtype, req_h, req_w

def preprocess_image(path: str, size_hw: Tuple[int,int], layout: str, dtype: str):
    """
    Load image -> RGB -> resize to (H,W) -> to dtype -> layout [1,H,W,C] or [1,C,H,W].
    float32 scaled to 0..1; uint8 kept 0..255.
    """
    H, W = size_hw
    img = Image.open(path).convert("RGB").resize((W, H), RESAMPLE_BICUBIC)
    arr = np.asarray(img)
    if dtype == "float32":
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)

    if layout == "NHWC":
        return arr[None, ...]  # [1,H,W,C]
    else:
        return np.transpose(arr, (2, 0, 1))[None, ...]  # [1,C,H,W]

def random_batch(size_hw: Tuple[int,int], layout: str, dtype: str):
    H, W = size_hw
    if dtype == "float32":
        if layout == "NHWC":
            return np.random.rand(1, H, W, 3).astype(np.float32)
        else:
            return np.random.rand(1, 3, H, W).astype(np.float32)
    else:
        if layout == "NHWC":
            return np.random.randint(0, 256, (1, H, W, 3), dtype=np.uint8)
        else:
            return np.random.randint(0, 256, (1, 3, H, W), dtype=np.uint8)

# ---------- Calibration reader ----------

class ImgCalibReader(CalibrationDataReader):
    def __init__(self, input_name, images, layout, dtype, size_hw: Tuple[int,int], n_samples=200):
        self.input_name = input_name
        self.layout = layout
        self.dtype = dtype
        self.size_hw = size_hw  # (H,W)

        images = list(images)
        random.shuffle(images)

        self.samples = []
        for p in images:
            try:
                self.samples.append(preprocess_image(p, size_hw, layout, dtype))
            except Exception:
                pass
            if len(self.samples) >= n_samples:
                break
        while len(self.samples) < n_samples:
            self.samples.append(random_batch(self.size_hw, self.layout, self.dtype))
        self._it = iter(self.samples)

    def get_next(self):
        try:
            return {self.input_name: next(self._it)}
        except StopIteration:
            return None

# ---------- Quantization ----------

def quantize_one(
    src,
    dst,
    calib_imgs,
    default_size=256,
    static=True,
    per_channel=True,
    percentile=None,   # e.g. 99.9 for Percentile; else MinMax
    qoperator=True,
    op_types=("Conv", "Gemm", "MatMul", "Add"),
):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Detect shape/dtype and input size
    _m, input_name, layout, dtype, req_h, req_w = get_input_info(src)

    # Decide size (H,W): use required if present, else default
    if (req_h and req_w) and (req_h > 0 and req_w > 0):
        H, W = req_h, req_w
    else:
        H = W = int(default_size)

    print(f"  └─ input='{input_name}' layout={layout} dtype={dtype} size={H}x{W}")

    if static:
        method = CalibrationMethod.Percentile if percentile is not None else CalibrationMethod.MinMax
        reader = ImgCalibReader(input_name, calib_imgs, layout, dtype, (H, W), n_samples=200)
        quantize_static(
            model_input=src,
            model_output=dst,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator if qoperator else QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=False,
            calibrate_method=method,
            op_types_to_quantize=list(op_types),
        )
    else:
        quantize_dynamic(
            model_input=src,
            model_output=dst,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            op_types_to_quantize=["Gemm", "MatMul", "Conv"],
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="public/models", help="input models dir")
    parser.add_argument("--dst", default="public/models-int8", help="output models dir")
    parser.add_argument("--imgdir", default="public/img", help="calibration images dir")
    parser.add_argument("--size", type=int, default=256, help="fallback size if model dims are dynamic")
    parser.add_argument("--dynamic", action="store_true", help="use dynamic quantization (no calibration)")
    parser.add_argument("--qdq", action="store_true", help="emit QDQ instead of QOperator")
    parser.add_argument("--percentile", type=float, default=None, help="e.g., 99.9 to use Percentile calibration")
    parser.add_argument("--per_channel", action="store_true", default=True)
    parser.add_argument("--ops", default="Conv,Gemm,MatMul,Add", help="comma-separated list of op types to quantize")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    models = sorted(glob.glob(os.path.join(args.src, "*.onnx")))
    imgs = []
    if os.path.isdir(args.imgdir):
        imgs = [p for p in glob.glob(os.path.join(args.imgdir, "*.*")) if not os.path.isdir(p)]

    os.makedirs(args.dst, exist_ok=True)
    op_types = tuple([s.strip() for s in args.ops.split(",") if s.strip()])

    for mpath in models:
        name = os.path.basename(mpath)
        out = os.path.join(args.dst, name.replace(".onnx", ".int8.onnx"))
        print(f"[quantize] {name} -> {os.path.basename(out)}")
        quantize_one(
            mpath,
            out,
            calib_imgs=imgs,
            default_size=args.size,
            static=not args.dynamic,
            per_channel=args.per_channel,
            percentile=args.percentile,
            qoperator=not args.qdq,
            op_types=op_types,
        )

if __name__ == "__main__":
    main()
