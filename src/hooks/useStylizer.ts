"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { PRESETS, RESIZE_MAX, PresetKey, isFNS } from "@/constants/presets";
import { createSession } from "@/lib/ort";
import {
  RangeMode,
  loadImage,
  rasterize,
  letterboxTo,
  runAutoLayout,
  type PostMode,
} from "@/lib/onnx";
import manifest from "@/constants/model-manifest.json" assert { type: "json" };

type StylizerState = {
  modelKey: PresetKey;
  setModelKey: (k: PresetKey) => void;

  imgUrl: string;
  pickImage: (f: File) => void;

  status: string;
  ready: boolean;
  isRunning: boolean;
  lastMs: number | null;

  dlUrl: string | null;
  run: () => Promise<void>;

  outCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  scratchRef: React.RefObject<HTMLCanvasElement | null>;

  strength: number; // 0..100
  setStrength: (v: number) => void;
};

type ManifestEntry = {
  input: string;
  layout: "NCHW" | "NHWC" | "unknown";
  H: number | null;
  W: number | null;
  dtype: string;
  ir_version: number;
  opset: number;
};

type CacheEntry = {
  modelKey: PresetKey;
  imgUrl: string;
  base: Uint8ClampedArray; // preprocessed original at W×H
  stylized: Uint8ClampedArray; // model output RGBA at W×H
  W: number;
  H: number;
  ms: number; // timing from last inference
  layout: "NHWC" | "NCHW";
};

function specFor(publicPath: string): ManifestEntry | null {
  const key = publicPath.replace(/^\/?models\//, "");
  return (manifest as Record<string, ManifestEntry | undefined>)[key] ?? null;
}

/** Linear blend: out = (1 - t) * base + t * stylized; t in [0,1]. */
function mixRGBA(
  base: Uint8ClampedArray,
  stylized: Uint8ClampedArray,
  amount01: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(stylized.length);
  const t = Math.min(1, Math.max(0, amount01));
  const u = 1 - t;
  for (let i = 0; i < stylized.length; i += 4) {
    out[i] = u * base[i] + t * stylized[i];
    out[i + 1] = u * base[i + 1] + t * stylized[i + 1];
    out[i + 2] = u * base[i + 2] + t * stylized[i + 2];
    out[i + 3] = 255;
  }
  return out;
}

export function useOnnxStylizer(
  initial: { modelKey?: PresetKey } = {},
): StylizerState {
  const [modelKey, setModelKey] = useState<PresetKey>(
    initial.modelKey ?? "ghibli",
  );
  const [status, setStatus] = useState<string>("Pick a style and an image.");
  const [ready, setReady] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [lastMs, setLastMs] = useState<number | null>(null);
  const [strength, setStrength] = useState<number>(100); // default full style

  const [imgUrl, setImgUrl] = useState<string>("");
  const imgUrlPrev = useRef<string | null>(null);

  const [dlUrl, setDlUrl] = useState<string | null>(null);
  const dlUrlPrev = useRef<string | null>(null);

  const outCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);

  const sessionRef = useRef<Awaited<ReturnType<typeof createSession>> | null>(
    null,
  );
  const inputNameRef = useRef<string>("");

  /** Cache only the most recent (modelKey + imgUrl) inference. */
  const lastResultRef = useRef<CacheEntry | null>(null);

  // revoke blob url on replace/unmount
  useEffect(
    () => () => {
      if (dlUrlPrev.current) URL.revokeObjectURL(dlUrlPrev.current);
    },
    [],
  );
  // revoke previous image url on unmount
  useEffect(
    () => () => {
      if (imgUrlPrev.current) URL.revokeObjectURL(imgUrlPrev.current);
    },
    [],
  );

  // clear cache when model changes
  useEffect(() => {
    lastResultRef.current = null;
    if (dlUrl) {
      URL.revokeObjectURL(dlUrl);
      setDlUrl(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelKey]);

  // load model on preset change
  useEffect(() => {
    const preset = PRESETS[modelKey];
    let canceled = false;
    (async () => {
      try {
        setReady(false);
        setStatus(`Loading ${preset.label}…`);
        const s = await createSession(preset.file);
        if (canceled) return;
        sessionRef.current = s;
        inputNameRef.current = s.inputNames[0] ?? "";
        setReady(true);
        setStatus(`Ready: ${preset.label}`);
      } catch (e) {
        sessionRef.current = null;
        setStatus(`Model load failed: ${getErrorMessage(e)}`);
      } finally {
        setLastMs(null);
      }
    })();
    return () => {
      canceled = true;
    };
  }, [modelKey]); // PRESETS is static

  const pickImage = useCallback(
    (f: File) => {
      const url = URL.createObjectURL(f);
      if (imgUrlPrev.current) URL.revokeObjectURL(imgUrlPrev.current);
      imgUrlPrev.current = url;
      setImgUrl(url);
      setLastMs(null);
      lastResultRef.current = null;
      if (dlUrlPrev.current) {
        URL.revokeObjectURL(dlUrlPrev.current);
        dlUrlPrev.current = null;
      }
      setDlUrl(null);
    },
    [dlUrl],
  );

  /** Draw a given RGBA buffer into output canvas and refresh the download URL. */
  const drawToCanvasAndSetDownload = useCallback(
    async (rgba: Uint8ClampedArray, W: number, H: number) => {
      const c = outCanvasRef.current;
      if (!c) {
        setStatus("Error: output canvas not available");
        return;
      }
      const ctx = c.getContext("2d");
      if (!ctx) {
        setStatus("Error: 2D context not available");
        return;
      }
      c.width = W;
      c.height = H;
      const imgData = ctx.createImageData(W, H);
      imgData.data.set(rgba);
      ctx.putImageData(imgData, 0, 0);

      const url = await canvasToBlobURL(c);
      if (dlUrlPrev.current) URL.revokeObjectURL(dlUrlPrev.current);
      dlUrlPrev.current = url;
      setDlUrl(url);
    },
    [outCanvasRef],
  );

  /** Re-blend from cache when strength changes (no inference). */
  useEffect(() => {
    const entry = lastResultRef.current;
    if (!entry) return;
    if (entry.modelKey !== modelKey || entry.imgUrl !== imgUrl) return;

    const s01 = Math.max(0, Math.min(1, strength / 100));
    const blended =
      s01 >= 1
        ? entry.stylized
        : s01 <= 0
          ? entry.base
          : mixRGBA(entry.base, entry.stylized, s01);

    // don't change lastMs; just redraw and keep status concise
    void drawToCanvasAndSetDownload(blended, entry.W, entry.H);
    setStatus(`Strength: ${strength}%`);
  }, [strength, modelKey, imgUrl, drawToCanvasAndSetDownload]);

  const run = useCallback(async () => {
    if (!ready || !imgUrl || !sessionRef.current || isRunning) return;

    try {
      setIsRunning(true);
      setStatus("Preparing image…");

      const img = await loadImage(imgUrl);
      const scratch = scratchRef.current;
      if (!scratch) {
        setStatus("Error: scratch canvas not available");
        return;
      }

      const preset = PRESETS[modelKey];
      const fns = isFNS(modelKey);
      const spec = specFor(preset.file);

      // Preprocess size: FNS → exact 224×224 letterbox; AnimeGAN → longest side 512
      const pre = fns
        ? letterboxTo(img, scratch, spec?.W ?? 224, spec?.H ?? 224)
        : rasterize(img, scratch, RESIZE_MAX);

      // Input/output conventions
      const inputRange: RangeMode = fns ? "0to1" : "m1to1";
      const postMode: PostMode = fns ? "minmax255" : "auto";

      // If we already have a cached result for this (modelKey, imgUrl) at same output W×H, skip inference.
      const cached = lastResultRef.current;
      if (cached && cached.modelKey === modelKey && cached.imgUrl === imgUrl) {
        // Ensure base matches cached W×H (rare mismatch)
        let baseAtWH = pre.rgba;
        if (cached.W !== pre.width || cached.H !== pre.height) {
          if (fns) {
            baseAtWH = letterboxTo(img, scratch, cached.W, cached.H).rgba;
          } else {
            scratch.width = cached.W;
            scratch.height = cached.H;
            const sctx = scratch.getContext("2d");
            if (!sctx) {
              setStatus("Error: 2D context not available");
              return;
            }
            sctx.clearRect(0, 0, cached.W, cached.H);
            sctx.drawImage(img, 0, 0, cached.W, cached.H);
            baseAtWH = sctx.getImageData(0, 0, cached.W, cached.H).data;
          }
          // update cache base to match its W×H for perfect blending next time
          lastResultRef.current = { ...cached, base: baseAtWH };
        }

        const s01 = Math.max(0, Math.min(1, strength / 100));
        const blended =
          s01 >= 1
            ? cached.stylized
            : s01 <= 0
              ? (lastResultRef.current?.base ?? pre.rgba)
              : mixRGBA(
                  lastResultRef.current?.base ?? pre.rgba,
                  cached.stylized,
                  s01,
                );

        await drawToCanvasAndSetDownload(blended, cached.W, cached.H);
        setLastMs(cached.ms);
        setStatus(`Cached blend • ${cached.ms.toFixed(1)} ms (prev)`);
        return;
      }

      setStatus("Stylizing…");
      const { rgbaOut, W, H, layoutUsed, ms } = await runAutoLayout(
        sessionRef.current,
        inputNameRef.current,
        pre.rgba,
        pre.width,
        pre.height,
        inputRange,
        postMode,
      );

      // Ensure base matches output size for blending & caching
      let baseAtWH = pre.rgba;
      if (W !== pre.width || H !== pre.height) {
        if (fns) {
          baseAtWH = letterboxTo(img, scratch, W, H).rgba;
        } else {
          scratch.width = W;
          scratch.height = H;
          const sctx = scratch.getContext("2d");
          if (!sctx) {
            setStatus("Error: 2D context not available");
            return;
          }
          sctx.clearRect(0, 0, W, H);
          sctx.drawImage(img, 0, 0, W, H);
          baseAtWH = sctx.getImageData(0, 0, W, H).data;
        }
      }

      // Cache for future strength-only updates
      lastResultRef.current = {
        modelKey,
        imgUrl,
        base: baseAtWH,
        stylized: rgbaOut,
        W,
        H,
        ms,
        layout: layoutUsed,
      };

      // Blend with current strength and draw
      const s01 = Math.max(0, Math.min(1, strength / 100));
      const blended =
        s01 >= 1
          ? rgbaOut
          : s01 <= 0
            ? baseAtWH
            : mixRGBA(baseAtWH, rgbaOut, s01);

      await drawToCanvasAndSetDownload(blended, W, H);
      setLastMs(ms);
      setStatus(`Done in ${ms.toFixed(1)} ms (${layoutUsed})`);
    } catch (e) {
      setStatus(`Error: ${getErrorMessage(e)}`);
    } finally {
      setIsRunning(false);
    }
  }, [
    ready,
    imgUrl,
    isRunning,
    modelKey,
    strength,
    drawToCanvasAndSetDownload,
  ]);

  return {
    modelKey,
    setModelKey,
    imgUrl,
    pickImage,
    status,
    ready,
    isRunning,
    lastMs,
    dlUrl,
    run,
    outCanvasRef,
    scratchRef,
    strength,
    setStrength,
  };
}

function getErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

async function canvasToBlobURL(c: HTMLCanvasElement, type = "image/png") {
  const blob = await new Promise<Blob>((resolve, reject) => {
    c.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob returned null"))),
      type,
    );
  });
  return URL.createObjectURL(blob);
}
