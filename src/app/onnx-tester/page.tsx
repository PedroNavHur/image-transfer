"use client";

import * as ort from "onnxruntime-web";
import { useEffect, useRef, useState } from "react";

export default function OnnxTester() {
  const [modelUrl, setModelUrl] = useState<string>("/models/animeganv3_arcane.onnx");
  const [imgUrl, setImgUrl] = useState<string>("");
  const [range, setRange] = useState<"0to1" | "m1to1">("0to1");
  const [resizeMax, setResizeMax] = useState<number>(512); // many AnimeGAN exports prefer 512
  const [status, setStatus] = useState<string>("Pick a model and an image.");
  const [lastMs, setLastMs] = useState<number | null>(null);
  const [ready, setReady] = useState(false);

  const outCanvasRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const inputNameRef = useRef<string>("");

  // ORT setup: use CDN for WASM binaries (no headers needed)
  useEffect(() => {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
  }, []);

  // Load model whenever modelUrl changes
  useEffect(() => {
    let canceled = false;
    (async () => {
      setReady(false);
      setStatus("Loading model…");
      try {
        const s = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        if (canceled) return;
        sessionRef.current = s;
        inputNameRef.current = s.inputNames[0];
        const dims =
          // try to surface dims if available
          ((s as any).inputMetadata?.[s.inputNames[0]]?.dimensions ??
            (s as any).inputMetadata?.[s.inputNames[0]]?.dims ??
            null) as number[] | null;
        setStatus(
          `Model ready: ${s.inputNames[0]} → ${s.outputNames[0]}${
            Array.isArray(dims) ? ` | dims: [${dims.join(", ")}]` : ""
          }`
        );
        setReady(true);
      } catch (e: any) {
        sessionRef.current = null;
        setStatus(`Model load failed: ${e?.message ?? String(e)}`);
      }
    })();
    return () => {
      canceled = true;
    };
  }, [modelUrl]);

  const onPickModel = (f: File) => setModelUrl(URL.createObjectURL(f));
  const onPickImage = (f: File) => setImgUrl(URL.createObjectURL(f));

  const run = async () => {
    if (!ready || !imgUrl || !sessionRef.current) return;
    try {
      setStatus("Preparing image…");
      const img = await loadImage(imgUrl);

      // Rasterize (optionally resize)
      const { width, height, rgba } = rasterize(img, resizeMax);

      // Prepare normalization
      const plane = width * height;
      const to01 = (v: number) => v / 255;
      const toM11 = (v: number) => v / 127.5 - 1.0;
      const norm = range === "m1to1" ? toM11 : to01;

      // Build both layouts (we'll try NHWC first, then NCHW fallback)
      const xNHWC = new Float32Array(plane * 3);
      for (let i = 0, p = 0; i < rgba.length; i += 4, p += 3) {
        xNHWC[p] = norm(rgba[i]);       // R
        xNHWC[p + 1] = norm(rgba[i+1]); // G
        xNHWC[p + 2] = norm(rgba[i+2]); // B
      }
      const xNCHW = new Float32Array(3 * plane);
      for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
        xNCHW[p] = norm(rgba[i]);             // R
        xNCHW[p + plane] = norm(rgba[i+1]);   // G
        xNCHW[p + 2 * plane] = norm(rgba[i+2]); // B
      }

      const feedsNHWC: Record<string, ort.Tensor> = {
        [inputNameRef.current]: new ort.Tensor("float32", xNHWC, [1, height, width, 3]),
      };
      const feedsNCHW: Record<string, ort.Tensor> = {
        [inputNameRef.current]: new ort.Tensor("float32", xNCHW, [1, 3, height, width]),
      };

      setStatus("Running (trying NHWC) …");
      let outputs: Record<string, ort.Tensor> | null = null;
      let layoutUsed: "NHWC" | "NCHW" = "NHWC";

      const t0 = performance.now();
      try {
        outputs = await sessionRef.current.run(feedsNHWC);
      } catch (_e) {
        setStatus("NHWC failed; retrying NCHW…");
        outputs = await sessionRef.current.run(feedsNCHW);
        layoutUsed = "NCHW";
      }
      const t1 = performance.now();
      setLastMs(t1 - t0);

      const outName = sessionRef.current.outputNames[0];
      const out = outputs[outName];
      const outRGBA = tensorToRgba(out);

      // Draw to canvas
      const c = outCanvasRef.current!;
      const { W, H } = getTensorWH(out, width, height);
      c.width = W; c.height = H;
      c.getContext("2d")!.putImageData(new ImageData(outRGBA, W, H), 0, 0);

      setStatus(`Done in ${(t1 - t0).toFixed(1)} ms (layout ${layoutUsed})`);
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
  };

  return (
    <main className="min-h-dvh bg-neutral-900 text-neutral-100">
      <div className="mx-auto w-full max-w-6xl space-y-6 p-6">
        <h1 className="text-2xl font-semibold">AnimeGANv3 (ONNX) — Next.js + Tailwind</h1>

        <div className="flex flex-wrap items-end gap-4 rounded-xl border border-neutral-800 bg-neutral-950 p-4">
          <div className="space-y-1">
            <div className="text-xs text-neutral-400">Model (.onnx)</div>
            <input
              type="file"
              accept=".onnx"
              onChange={(e) => e.target.files?.[0] && onPickModel(e.target.files[0])}
              className="block w-64 text-sm file:mr-3 file:rounded-md file:border file:border-neutral-700 file:bg-neutral-800 file:px-3 file:py-1.5 file:text-neutral-100"
            />
          </div>

          <div className="space-y-1">
            <div className="text-xs text-neutral-400">Image</div>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && onPickImage(e.target.files[0])}
              className="block w-64 text-sm file:mr-3 file:rounded-md file:border file:border-neutral-700 file:bg-neutral-800 file:px-3 file:py-1.5 file:text-neutral-100"
            />
          </div>

          <div className="space-y-1">
            <div className="text-xs text-neutral-400">Resize max side (px)</div>
            <input
              type="number"
              min={0}
              value={resizeMax}
              onChange={(e) => setResizeMax(parseInt(e.target.value || "0", 10))}
              className="w-28 rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm"
            />
          </div>

          <div className="space-y-1">
            <div className="text-xs text-neutral-400">Input range</div>
            <select
              value={range}
              onChange={(e) => setRange(e.target.value as any)}
              className="rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm"
            >
              <option value="0to1">[0,1]</option>
              <option value="m1to1">[-1,1]</option>
            </select>
          </div>

          <button
            onClick={run}
            disabled={!ready || !imgUrl}
            className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-sm font-medium hover:bg-neutral-750 disabled:opacity-50"
          >
            Run
          </button>

          <span className="text-xs text-neutral-400">{status}</span>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
            <h3 className="mb-2 text-sm font-medium text-neutral-200">Original</h3>
            {imgUrl ? (
              <img src={imgUrl} alt="" className="w-full rounded-lg border border-neutral-800 bg-neutral-900" />
            ) : (
              <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-neutral-800 text-neutral-500">
                No image selected
              </div>
            )}
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
            <h3 className="mb-2 text-sm font-medium text-neutral-200">Output</h3>
            <canvas ref={outCanvasRef} className="w-full rounded-lg border border-neutral-800 bg-neutral-900" />
            {lastMs !== null && (
              <div className="mt-2 text-xs text-neutral-400">Inference time: {lastMs.toFixed(1)} ms</div>
            )}
          </div>
        </div>

        <canvas ref={(c) => (scratchRef.current = c)} className="hidden" />
      </div>
    </main>
  );

  // -------- helpers --------
  function loadImage(src: string) {
    return new Promise<HTMLImageElement>((resolve) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.src = src;
    });
  }

  function rasterize(img: HTMLImageElement, maxSide: number) {
    const scratch = scratchRef.current!;
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

  function getTensorWH(out: ort.Tensor, fallbackW: number, fallbackH: number) {
    const d = out.dims;
    if (d.length === 4 && d[1] === 3) return { W: d[3], H: d[2] }; // NCHW
    if (d.length === 4 && d[3] === 3) return { W: d[2], H: d[1] }; // NHWC
    return { W: fallbackW, H: fallbackH };
  }

  function tensorToRgba(out: ort.Tensor): Uint8ClampedArray {
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
      // fallback assume NCHW
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
}
