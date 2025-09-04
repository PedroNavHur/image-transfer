Image Transfer is a local, browser‑based image‑to‑image style transfer app that runs ONNX models via onnxruntime‑web. It supports quantized INT8 models with safe fallbacks to FP32.

## Run The App

- Dev server: `npm run dev` (Next.js 15 + Turbopack)
- Open: http://localhost:3000
- Use: pick a style and an image, adjust Strength slider.

## Models & Presets

- Families:
  - agan: AnimeGANv3, dynamic NHWC input (studio look styles: Ghibli, Hayao, Shinkai, Sketch)
  - fns: Fast Neural Style, fixed 224×224 NCHW (Mosaic, Candy, Udnie, Pointilism)
- Paths:
  - Quantized INT8: `public/models-int8/*.int8.onnx`
  - FP32 originals: `public/models-agan/*.onnx` (AnimeGAN) and `public/models/*.onnx` (FNS)

## INT8 vs FP32 Fallback

- The status line shows which model is active after load:
  - “Ready: <Style> (INT8)” → running the quantized model.
  - “Ready: <Style> (FP32 fallback)” → quantized model failed to load; using FP32.
- Console diagnostics:
  - Failures summarize as one line prefixed with `[ORT-LOAD-FAIL] { ... }`.
  - Enable verbose ORT logs only if needed: add `?ort_debug=1` to the URL or run `localStorage.setItem('ort_debug','1')` and reload.

## Quantization (Python)

Quantization scripts live in `python/`. We use gentle, WASM‑friendly settings and per‑family defaults.

- Environment (suggested):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install onnx onnxruntime onnxruntime-tools pillow numpy`

- Command (AnimeGAN, higher precision but stable):
  - `python python/quantize_models.py --src public/models-agan --dst public/models-int8 --only agan_ghibli.onnx,agan_hayao.onnx,agan_shinkai.onnx --samples 64 --size 192 --percentile 99.99`
  - Defaults used per family:
    - agan: QOperator (QLinearConv), Conv only, per‑tensor (per‑channel off), opset 13.
    - fns: QOperator (QLinearConv/QLinearMatMul), Conv + MatMul, per‑channel on, opset 13.

- Command (FNS styles):
  - `python python/quantize_models.py --src public/models --dst public/models-int8 --only fns_candy.onnx,fns_mosaic.onnx,fns_udnie.onnx,fns_pointilism.onnx --samples 64 --size 224 --percentile 99.9`

- Useful flags:
  - `--only` filter by basenames/globs (comma‑separated).
  - `--samples` calibration images to use (static only). More samples → better fidelity, slower.
  - `--size` calibration size. Larger → better fidelity, slower.
  - `--percentile` activation percentiles (e.g., 99.9). Higher reduces clipping.
  - `--force_per_channel` enable per‑channel weights even for AnimeGAN.
  - `--qdq` emit QDQ instead of QOperator (generally keep QOperator for WASM).

Notes:
- The script pre‑upgrades FP32 models to opset 13 before quantization for cleaner graphs.
- It may emit temporary `*.preopset.onnx` files in `public/models-int8/`; these can be deleted.

### NPM Script Shortcuts

- `npm run quant:agan` → AnimeGAN (all): samples=64, size=192, percentile=99.99
- `npm run quant:fns` → FNS (all): samples=64, size=224, percentile=99.9
- `npm run quant:ghibli` → Ghibli only (same settings as agan)
- `npm run quant:clean` → remove any `*.preopset.onnx` intermediates

## onnxruntime‑web Behavior

- Execution provider order: WebGPU (if available) then WASM.
- WASM uses SIMD and a single thread by default (no COOP/COEP needed).
- If an INT8 model fails to load under WASM, the app automatically falls back to the FP32 model and marks the status as “(FP32 fallback)”.

## Troubleshooting

- Noisy console: default logs are minimal. Turn on deep logs with `?ort_debug=1` only when requested.
- Share failures by copying the single `[ORT-LOAD-FAIL]` line from the console.
- If a specific model keeps falling back, try gentler quantization (reduce ops, turn off per‑channel, lower percentile) and re‑quantize that model with `--only`.
