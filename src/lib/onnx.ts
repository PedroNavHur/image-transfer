"use client";
import type { InferenceSession, Tensor as ORTTensor } from "onnxruntime-web";
import { Tensor } from "onnxruntime-web"; // runtime constructor

// ------- image helpers -------
export function loadImage(src: string) {
  return new Promise<HTMLImageElement>(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

export function rasterize(
  img: HTMLImageElement,
  scratch: HTMLCanvasElement,
  maxSide: number
) {
  let w = img.naturalWidth;
  let h = img.naturalHeight;
  if (maxSide > 0 && Math.max(w, h) > maxSide) {
    const s = maxSide / Math.max(w, h);
    w = Math.round(w * s);
    h = Math.round(h * s);
  }
  scratch.width = w;
  scratch.height = h;
  const ctx = scratch.getContext("2d")!;
  ctx.drawImage(img, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  return { width: w, height: h, rgba: data };
}

export type RangeMode = "0to1" | "m1to1";

export function buildInputs(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  range: RangeMode
) {
  const plane = width * height;
  const to01 = (v: number) => v / 255;
  const toM11 = (v: number) => v / 127.5 - 1.0;
  const norm = range === "m1to1" ? toM11 : to01;

  // NHWC
  const xNHWC = new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 3) {
    xNHWC[p] = norm(rgba[i]);
    xNHWC[p + 1] = norm(rgba[i + 1]);
    xNHWC[p + 2] = norm(rgba[i + 2]);
  }

  // NCHW
  const xNCHW = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    xNCHW[p] = norm(rgba[i]);
    xNCHW[p + plane] = norm(rgba[i + 1]);
    xNCHW[p + 2 * plane] = norm(rgba[i + 2]);
  }

  return { xNHWC, xNCHW };
}

/** Width/height from tensor dims, fallback to provided values if ambiguous. */
export function getTensorWH(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number
) {
  const d = out.dims;

  // 4D: [N,C,H,W] or [N,H,W,3]
  if (d.length === 4) {
    if (d[1] === 3) return { W: d[3], H: d[2] }; // NCHW
    if (d[3] === 3) return { W: d[2], H: d[1] }; // NHWC
  }

  // 3D: [C,H,W] or [H,W,3]
  if (d.length === 3) {
    if (d[0] === 3) return { W: d[2], H: d[1] }; // CHW
    if (d[2] === 3) return { W: d[1], H: d[0] }; // HWC
  }

  // Ambiguous or non-RGB
  return { W: fallbackW, H: fallbackH };
}

/** Convert output tensor (NCHW/NHWC/CHW/HWC; [0,1] or [-1,1]) to RGBA. */
export function tensorToRgba(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number
): Uint8ClampedArray {
  const dims = out.dims;
  const data = out.data as Float32Array;

  // Determine layout and W/H
  const { W, H } = getTensorWH(out, fallbackW, fallbackH);

  // Indexer for reading channels regardless of layout
  let get: (c: 0 | 1 | 2, p: number) => number;

  if (dims.length === 4 && dims[1] === 3) {
    // NCHW
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  } else if (dims.length === 4 && dims[3] === 3) {
    // NHWC
    get = (c, p) => data[p * 3 + c];
  } else if (dims.length === 3 && dims[0] === 3) {
    // CHW
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  } else if (dims.length === 3 && dims[2] === 3) {
    // HWC
    get = (c, p) => data[p * 3 + c];
  } else {
    // Fallback: assume NCHW with provided W/H
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  }

  // Decide mapping [0,1] vs [-1,1] via simple variance heuristic
  const as01 = (v: number) => clamp255(v * 255);
  const asM11 = (v: number) => clamp255((v + 1) * 127.5);
  const map = pickMap(get, W, H, as01, asM11);

  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let p = 0; p < W * H; p++) {
    rgba[4 * p] = map(get(0, p));
    rgba[4 * p + 1] = map(get(1, p));
    rgba[4 * p + 2] = map(get(2, p));
    rgba[4 * p + 3] = 255;
  }
  return rgba;
}

function pickMap(
  get: (c: 0 | 1 | 2, p: number) => number,
  W: number,
  H: number,
  a: (v: number) => number,
  b: (v: number) => number
) {
  const plane = W * H;
  const variance = (map: (v: number) => number) => {
    let s = 0,
      s2 = 0,
      n = 0;
    for (let p = 0; p < Math.min(plane, 500); p += 5) {
      const m = (map(get(0, p)) + map(get(1, p)) + map(get(2, p))) / 3;
      s += m;
      s2 += m * m;
      n++;
    }
    const mean = s / n;
    return s2 / n - mean * mean;
  };
  return variance(b) > variance(a) ? b : a;
}
function clamp255(v: number) {
  return Math.max(0, Math.min(255, v));
}

// ------- core runner (auto NHWC→NCHW) -------
export async function runAutoLayout(
  session: InferenceSession,
  inputName: string,
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  range: RangeMode
) {
  const { xNHWC, xNCHW } = buildInputs(rgba, width, height, range);

  const feedsNHWC = {
    [inputName]: new Tensor("float32", xNHWC, [1, height, width, 3]),
  };
  const feedsNCHW = {
    [inputName]: new Tensor("float32", xNCHW, [1, 3, height, width]),
  };

  const t0 = performance.now();

  let outputs: Awaited<ReturnType<InferenceSession["run"]>>;
  let layoutUsed: "NHWC" | "NCHW" = "NHWC";

  try {
    outputs = await session.run(feedsNHWC);
  } catch {
    outputs = await session.run(feedsNCHW);
    layoutUsed = "NCHW";
  }
  const t1 = performance.now();

  const outName = session.outputNames[0];
  const out = outputs[outName] as ORTTensor;

  // pass original input size as explicit fallback (no any-casts)
  const rgbaOut = tensorToRgba(out, width, height);
  const { W, H } = getTensorWH(out, width, height);

  return { rgbaOut, W, H, layoutUsed, ms: t1 - t0 };
}
