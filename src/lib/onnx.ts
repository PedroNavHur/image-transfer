"use client";
import type { InferenceSession, Tensor as ORTTensor } from "onnxruntime-web";
import { Tensor } from "onnxruntime-web";

// ---------- image helpers ----------
export function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

export function rasterize(
  img: HTMLImageElement,
  scratch: HTMLCanvasElement,
  maxSide: number,
) {
  let w = img.naturalWidth,
    h = img.naturalHeight;
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

// NEW: exact WxH with letterbox (centered, no stretch)
export function letterboxTo(
  img: HTMLImageElement,
  scratch: HTMLCanvasElement,
  W: number,
  H: number,
  fill = "#f2f2f2",
) {
  scratch.width = W;
  scratch.height = H;
  const ctx = scratch.getContext("2d")!;
  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, W, H);
  const s = Math.min(W / img.naturalWidth, H / img.naturalHeight);
  const dw = Math.round(img.naturalWidth * s);
  const dh = Math.round(img.naturalHeight * s);
  const dx = Math.floor((W - dw) / 2),
    dy = Math.floor((H - dh) / 2);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, dx, dy, dw, dh);
  const { data } = ctx.getImageData(0, 0, W, H);
  return { width: W, height: H, rgba: data };
}

export type RangeMode = "0to1" | "m1to1";

export function buildInputs(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  range: RangeMode,
) {
  const plane = width * height;
  const to01 = (v: number) => v / 255;
  const toM11 = (v: number) => v / 127.5 - 1.0;
  const norm = range === "m1to1" ? toM11 : to01;

  const xNHWC = new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 3) {
    xNHWC[p] = norm(rgba[i]);
    xNHWC[p + 1] = norm(rgba[i + 1]);
    xNHWC[p + 2] = norm(rgba[i + 2]);
  }

  const xNCHW = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    xNCHW[p] = norm(rgba[i]);
    xNCHW[p + plane] = norm(rgba[i + 1]);
    xNCHW[p + 2 * plane] = norm(rgba[i + 2]);
  }

  return { xNHWC, xNCHW };
}

// ---------- tensor shape helpers ----------
export function getTensorWH(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number,
) {
  const d = out.dims;
  if (d.length === 4 && d[1] === 3) return { W: d[3], H: d[2] }; // NCHW
  if (d.length === 4 && d[3] === 3) return { W: d[2], H: d[1] }; // NHWC
  if (d.length === 3 && d[0] === 3) return { W: d[2], H: d[1] }; // CHW
  if (d.length === 3 && d[2] === 3) return { W: d[1], H: d[0] }; // HWC
  return { W: fallbackW, H: fallbackH };
}

// ---------- post-process ----------
function clamp255(v: number) {
  return Math.max(0, Math.min(255, v));
}

export function tensorToRgba_auto(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number,
): Uint8ClampedArray {
  const dims = out.dims;
  const data = out.data as Float32Array;
  const { W, H } = getTensorWH(out, fallbackW, fallbackH);

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
  } else {
    // HWC or fallback
    get = (c, p) => data[p * 3 + c];
  }

  // choose mapping between [0,1] and [-1,1]
  const as01 = (v: number) => clamp255(v * 255);
  const asM11 = (v: number) => clamp255((v + 1) * 127.5);
  const rgba = new Uint8ClampedArray(W * H * 4);

  // simple variance-based pick
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
  const map = variance(asM11) > variance(as01) ? asM11 : as01;

  for (let p = 0; p < plane; p++) {
    rgba[4 * p + 0] = map(get(0, p));
    rgba[4 * p + 1] = map(get(1, p));
    rgba[4 * p + 2] = map(get(2, p));
    rgba[4 * p + 3] = 255;
  }
  return rgba;
}

// NEW: robust min-max mapping (good for FNS which may output raw 0..255 or arbitrary range)
export function tensorToRgba_minmax(
  out: ORTTensor,
  fallbackW: number,
  fallbackH: number,
): Uint8ClampedArray {
  const dims = out.dims;
  const data = out.data as Float32Array;
  const { W, H } = getTensorWH(out, fallbackW, fallbackH);

  let get: (c: 0 | 1 | 2, p: number) => number;
  if (dims.length === 4 && dims[1] === 3) {
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  } else if (dims.length === 4 && dims[3] === 3) {
    get = (c, p) => data[p * 3 + c];
  } else if (dims.length === 3 && dims[0] === 3) {
    const plane = W * H;
    get = (c, p) => data[p + c * plane];
  } else {
    get = (c, p) => data[p * 3 + c];
  }

  // detect quick cases first
  let min = Infinity,
    max = -Infinity;
  const plane = W * H;
  for (let p = 0; p < plane; p++) {
    const r = get(0, p),
      g = get(1, p),
      b = get(2, p);
    if (r < min) min = r;
    if (g < min) min = g;
    if (b < min) min = b;
    if (r > max) max = r;
    if (g > max) max = g;
    if (b > max) max = b;
  }

  const rgba = new Uint8ClampedArray(W * H * 4);
  // If it already looks like [0,1]
  if (max <= 1.2 && min >= -0.2) {
    for (let p = 0; p < plane; p++) {
      rgba[4 * p + 0] = clamp255(get(0, p) * 255);
      rgba[4 * p + 1] = clamp255(get(1, p) * 255);
      rgba[4 * p + 2] = clamp255(get(2, p) * 255);
      rgba[4 * p + 3] = 255;
    }
    return rgba;
  }
  // If ~[-1,1]
  if (max <= 1.1 && min >= -1.1) {
    for (let p = 0; p < plane; p++) {
      rgba[4 * p + 0] = clamp255((get(0, p) * 0.5 + 0.5) * 255);
      rgba[4 * p + 1] = clamp255((get(1, p) * 0.5 + 0.5) * 255);
      rgba[4 * p + 2] = clamp255((get(2, p) * 0.5 + 0.5) * 255);
      rgba[4 * p + 3] = 255;
    }
    return rgba;
  }
  // General case: min-max to [0,255]
  const gain = max === min ? 1 : 255 / (max - min);
  for (let p = 0; p < plane; p++) {
    rgba[4 * p + 0] = clamp255((get(0, p) - min) * gain);
    rgba[4 * p + 1] = clamp255((get(1, p) - min) * gain);
    rgba[4 * p + 2] = clamp255((get(2, p) - min) * gain);
    rgba[4 * p + 3] = 255;
  }
  return rgba;
}

// ---------- runner ----------
export type PostMode = "auto" | "minmax255";

export async function runAutoLayout(
  session: InferenceSession,
  inputName: string,
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  range: RangeMode,
  postMode: PostMode = "auto",
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
    outputs = await session.run(
      feedsNHWC as unknown as Record<string, ORTTensor>,
    );
  } catch {
    outputs = await session.run(
      feedsNCHW as unknown as Record<string, ORTTensor>,
    );
    layoutUsed = "NCHW";
  }
  const t1 = performance.now();

  const outName = session.outputNames[0];
  const out = outputs[outName];

  const { W, H } = getTensorWH(out, width, height);
  const rgbaOut =
    postMode === "minmax255"
      ? tensorToRgba_minmax(out, width, height)
      : tensorToRgba_auto(out, width, height);

  return { rgbaOut, W, H, layoutUsed, ms: t1 - t0 };
}
