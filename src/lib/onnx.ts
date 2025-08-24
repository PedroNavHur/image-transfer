"use client";
import type { InferenceSession, Tensor as ORTTensor } from "onnxruntime-web";
import { Tensor } from "onnxruntime-web"; // <-- value import (runtime constructor)

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

export function getTensorWH(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number
) {
  const d = out.dims;
  if (d.length === 4 && d[1] === 3) return { W: d[3], H: d[2] }; // NCHW
  if (d.length === 4 && d[3] === 3) return { W: d[2], H: d[1] }; // NHWC
  return { W: fallbackW, H: fallbackH };
}

export function tensorToRgba(out: ORTTensor): Uint8ClampedArray {
  const dims = out.dims;
  const data = out.data as Float32Array;

  let W: number, H: number;
  let get: (c: 0 | 1 | 2, p: number) => number;

  if (dims.length === 4 && dims[1] === 3) {
    // NCHW
    H = dims[2];
    W = dims[3];
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  } else if (dims.length === 4 && dims[3] === 3) {
    // NHWC
    H = dims[1];
    W = dims[2];
    get = (c, p) => data[p * 3 + c];
  } else {
    H = (out as any).height ?? 0;
    W = (out as any).width ?? 0;
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  }

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
  let outputs: Record<string, ORTTensor>;
  let layoutUsed: "NHWC" | "NCHW" = "NHWC";

  try {
    outputs = await session.run(feedsNHWC);
  } catch {
    outputs = await session.run(feedsNCHW);
    layoutUsed = "NCHW";
  }
  const t1 = performance.now();

  const outName = session.outputNames[0];
  const out = outputs[outName];
  const rgbaOut = tensorToRgba(out);
  const { W, H } = getTensorWH(out, width, height);

  return { rgbaOut, W, H, layoutUsed, ms: t1 - t0 };
}
