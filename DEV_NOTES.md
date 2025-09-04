Project: Image Transfer (ONNX Web)

Summary
- Issue: Recently quantized INT8 models intermittently failed to load in the browser (opaque WASM abort codes). Sketch model OK; others not.
- Fixes implemented:
  - Centralized ORT init in `src/lib/ort.ts` with controlled logging, SIMD on, single-thread WASM (no COOP/COEP required).
  - Robust session loader: try `webgpu+wasm/all` → `wasm/all` → `wasm/basic` → `wasm/disabled`; then auto‑fallback to FP32.
  - Structured console logs: concise `[ORT-LOAD-FAIL] { ... }` summary on failure, optional deep logs via `?ort_debug=1`.
  - UI status badge shows which model is active: `(INT8)` vs `(FP32 fallback)`.
  - Types tightened: removed `any` casts by extending `InferenceSession` with `__loadMeta` in a typed way.

Quantization Strategy (python/quantize_models.py)
- Family‑aware gentle defaults:
  - AnimeGAN (agan, NHWC dynamic): QOperator (QLinearConv), quantize Conv only, per‑tensor (per‑channel off), opset 13, Percentile calibration.
  - Fast Neural Style (fns, NCHW 224): QOperator (QLinearConv/QLinearMatMul), quantize Conv + MatMul, per‑channel on, opset 13, Percentile calibration.
- CLI improvements:
  - `--only` to target specific models (glob/basenames).
  - `--samples` (calibration set size), `--size` (calibration image size), `--percentile`.
  - `--force_per_channel` to override AnimeGAN per‑tensor default if you want higher fidelity.
  - `--qdq` available (prefer QOperator for WASM unless debugging).
- Opset handling: FP32 models pre‑upgraded to opset 13 before quantization; intermediate `*.preopset.onnx` files are cleaned up.

What’s Quantized Now
- AnimeGAN (INT8 working):
  - ghibli, hayao, shinkai
  - Settings used: `--samples 64 --size 192 --percentile 99.99`, Conv‑only, per‑tensor, QOperator, opset 13.
- FNS (INT8 working):
  - candy, mosaic, udnie, pointilism
  - Settings used: `--samples 64 --size 224 --percentile 99.9`, Conv+MatMul, per‑channel, QOperator, opset 13.

How to Verify in App
- Status line: shows “Ready: <Style> (INT8)” when quantized model is active; “(FP32 fallback)” otherwise.
- Console:
  - Filter for `ORT-LOAD`.
  - On load failure you get a single structured `[ORT-LOAD-FAIL] { ... }` line describing attempts.
  - Enable deep logs temporarily with `?ort_debug=1` or `localStorage.setItem('ort_debug','1')`.

Common Commands
- Dev server: `npm run dev` → http://localhost:3000
- Quantize AnimeGAN (all three):
  - `python python/quantize_models.py --src public/models-agan --dst public/models-int8 --only agan_ghibli.onnx,agan_hayao.onnx,agan_shinkai.onnx --samples 64 --size 192 --percentile 99.99`
- Quantize FNS (all four):
  - `python python/quantize_models.py --src public/models --dst public/models-int8 --only fns_candy.onnx,fns_mosaic.onnx,fns_udnie.onnx,fns_pointilism.onnx --samples 64 --size 224 --percentile 99.9`
- Try per‑channel on AnimeGAN (optional): add `--force_per_channel`.

Tuning Guidance
- Stability first (AnimeGAN): keep Conv‑only, per‑tensor. If stable and you want more fidelity, increase `--size` to 224, then try `--force_per_channel`.
- Fidelity knobs: increase `--samples` (32→64→128), `--size` (192→224), and `--percentile` (99.9→99.99) cautiously.
- If a specific INT8 model falls back to FP32: capture the single `[ORT-LOAD-FAIL]` line; try gentler quant (reduce ops, turn off per‑channel, lower percentile) with `--only`.

Code Touchpoints
- Loader: `src/lib/ort.ts` (init, loader, structured logs, metadata), `src/hooks/useStylizer.ts` (status tag, error messages).
- ONNX utilities: `src/lib/onnx.ts` (input builders, post‑processing; ORT env moved out).
- UI controls: `src/components/OnnXControls.tsx` (props renamed to `*Action` to avoid Next Server Action warnings).
- Presets/config: `src/constants/presets.ts`, `public/models-int8/*`.
- Docs: `README.md` updated with workflow and troubleshooting.

Next Steps (Optional)
- Add a script to clean intermediates: remove `*.preopset.onnx` after quantization.
- Add a simple NPM task to quantize per family (wrapping the Python commands).
- Consider COOP/COEP headers if enabling WASM multi‑threading for perf (currently single‑thread to avoid header requirements).
- If WebGPU is available, you can revisit executionProvider order to prefer it for speed.

Notes
- We intentionally keep ORT logs minimal. Use the debug flag only when diagnosing.
- FP32 fallback ensures UX continuity; status clearly indicates the mode.
