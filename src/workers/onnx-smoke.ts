// workers/onnx-smoke.ts
import * as ort from "onnxruntime-web";

type InitMsg = {
  type: "init";
  modelUrl: string;
  ep: "webgpu" | "wasm" | "auto";
};
type RunMsg = {
  type: "run";
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
  layout: "NCHW" | "NHWC";
  range: "0to1" | "minus1to1";
  fixedSize?: number | null; // e.g., 512 if model needs fixed H=W
};

let session: ort.InferenceSession | null = null;

self.onmessage = async (e: MessageEvent<InitMsg | RunMsg>) => {
  const msg = e.data;
  if (msg.type === "init") {
    const providers = msg.ep === "auto" ? ["webgpu", "wasm"] : [msg.ep];
    session = await ort.InferenceSession.create(msg.modelUrl, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    });
    (self as any).postMessage({
      type: "ready",
      inputs: session.inputNames,
      outputs: session.outputNames,
    });
    return;
  }

  if (msg.type === "run") {
    if (!session) {
      (self as any).postMessage({
        type: "error",
        error: "Session not initialized",
      });
      return;
    }

    const { rgba, width, height, layout, range, fixedSize } = msg;

    // Resize in-worker if model needs fixed size
    const [W, H] = fixedSize ? [fixedSize, fixedSize] : [width, height];
    let src = rgba;
    let sw = width;
    let sh = height;
    if (fixedSize && (width !== fixedSize || height !== fixedSize)) {
      // Simple nearest-neighbor resize on CPU (good enough for a smoke test)
      const resized = new Uint8ClampedArray(W * H * 4);
      for (let y = 0; y < H; y++) {
        const sy = Math.floor((y * sh) / H);
        for (let x = 0; x < W; x++) {
          const sx = Math.floor((x * sw) / W);
          const si = (sy * sw + sx) * 4;
          const di = (y * W + x) * 4;
          resized[di] = src[si];
          resized[di + 1] = src[si + 1];
          resized[di + 2] = src[si + 2];
          resized[di + 3] = 255;
        }
      }
      src = resized;
      sw = W;
      sh = H;
    }

    // Normalize
    const plane = sw * sh;
    const to0_1 = (v: number) => v / 255;
    const toM1_1 = (v: number) => v / 127.5 - 1.0;

    const conv = range === "0to1" ? to0_1 : toM1_1;

    let input: ort.Tensor;
    if (layout === "NCHW") {
      const chw = new Float32Array(3 * plane);
      for (let i = 0, p = 0; i < src.length; i += 4, p++) {
        chw[p] = conv(src[i]);
        chw[p + plane] = conv(src[i + 1]);
        chw[p + 2 * plane] = conv(src[i + 2]);
      }
      input = new ort.Tensor("float32", chw, [1, 3, sh, sw]);
    } else {
      // NHWC
      const nhwc = new Float32Array(plane * 3);
      for (let i = 0, p = 0; i < src.length; i += 4, p += 3) {
        nhwc[p] = conv(src[i]);
        nhwc[p + 1] = conv(src[i + 1]);
        nhwc[p + 2] = conv(src[i + 2]);
      }
      input = new ort.Tensor("float32", nhwc, [1, sh, sw, 3]);
    }

    const feeds: Record<string, ort.Tensor> = {};
    // Best-effort: try common input names
    const inName =
      session.inputNames.find(n =>
        ["input", "images", "x", "data"].includes(n.toLowerCase())
      ) || session.inputNames[0];
    feeds[inName] = input;

    const t0 = performance.now();
    const outputs = await session.run(feeds);
    const t1 = performance.now();

    const outName = session.outputNames[0];
    const out = outputs[outName] as ort.Tensor;
    const od = out.data as Float32Array;
    // Guess output layout from dims
    let outW = sw,
      outH = sh;
    if (out.dims.length === 4) {
      // [1,C,H,W] or [1,H,W,3]
      if (out.dims[1] === 3) {
        outH = out.dims[2];
        outW = out.dims[3];
      } // NCHW
      else {
        outH = out.dims[1];
        outW = out.dims[2];
      } // NHWC
    }
    const oplane = outW * outH;
    const rgbaOut = new Uint8ClampedArray(outW * outH * 4);

    // Assume output in [0,1] or [-1,1]. Auto-scale by clamping.
    const clamp255 = (v: number) => Math.max(0, Math.min(255, v));
    // Try both maps; pick the one with larger variance
    const as01 = (x: number) => clamp255(x * 255);
    const asM11 = (x: number) => clamp255((x + 1) * 127.5);

    // Detect channel order (NCHW vs NHWC) by dims
    let get = (c: 0 | 1 | 2, p: number) => {
      if (out.dims.length === 4 && out.dims[1] === 3) return od[p + c * oplane]; // NCHW
      if (out.dims.length === 4 && out.dims[3] === 3) return od[p * 3 + c]; // NHWC
      // Fallback: assume NCHW
      return od[p + c * oplane];
    };

    // Decide scaling by sampling few pixels
    const sample = (fn: (x: number) => number) => {
      let sum = 0,
        sum2 = 0,
        n = 0;
      for (let p = 0; p < Math.min(500, oplane); p += 5) {
        const r = fn(get(0, p));
        const g = fn(get(1, p));
        const b = fn(get(2, p));
        const m = (r + g + b) / 3;
        sum += m;
        sum2 += m * m;
        n++;
      }
      const mean = sum / n;
      const varc = sum2 / n - mean * mean;
      return varc;
    };
    const var01 = sample(as01),
      varM11 = sample(asM11);
    const map = varM11 > var01 ? asM11 : as01;

    for (let p = 0; p < oplane; p++) {
      rgbaOut[4 * p] = map(get(0, p));
      rgbaOut[4 * p + 1] = map(get(1, p));
      rgbaOut[4 * p + 2] = map(get(2, p));
      rgbaOut[4 * p + 3] = 255;
    }

    (self as any).postMessage(
      {
        type: "done",
        ms: t1 - t0,
        width: outW,
        height: outH,
        rgba: rgbaOut,
      },
      [rgbaOut.buffer]
    );
  }
};
