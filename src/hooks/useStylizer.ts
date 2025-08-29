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

function specFor(publicPath: string): ManifestEntry | null {
  const key = publicPath.replace(/^\/?models\//, "");
  return (manifest as Record<string, ManifestEntry | undefined>)[key] ?? null;
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

  const [imgUrl, setImgUrl] = useState<string>("");
  const imgUrlPrev = useRef<string | null>(null);

  const [dlUrl, setDlUrl] = useState<string | null>(null);

  const outCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);

  const sessionRef = useRef<Awaited<ReturnType<typeof createSession>> | null>(
    null,
  );
  const inputNameRef = useRef<string>("");

  // revoke blob url on replace/unmount
  useEffect(
    () => () => {
      if (dlUrl) URL.revokeObjectURL(dlUrl);
    },
    [dlUrl],
  );
  // revoke previous image url on unmount
  useEffect(
    () => () => {
      if (imgUrlPrev.current) URL.revokeObjectURL(imgUrlPrev.current);
    },
    [],
  );

  // clean up dlUrl when model changes
  useEffect(() => {
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
      if (dlUrl) {
        URL.revokeObjectURL(dlUrl);
        setDlUrl(null);
      }
    },
    [dlUrl],
  );

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

      // Size: FNS → exact 224×224 letterbox; AnimeGAN → max side 512
      const pre = fns
        ? letterboxTo(img, scratch, spec?.W ?? 224, spec?.H ?? 224)
        : rasterize(img, scratch, RESIZE_MAX);

      // Input range: FNS → [0,1]; AnimeGAN → [-1,1]
      const inputRange: RangeMode = fns ? "0to1" : "m1to1";
      // Post mode: FNS → minmax to [0,255]; AnimeGAN → auto
      const postMode: PostMode = fns ? "minmax255" : "auto";

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
      imgData.data.set(rgbaOut);
      ctx.putImageData(imgData, 0, 0);

      const url = await canvasToBlobURL(c);
      if (dlUrl) URL.revokeObjectURL(dlUrl);
      setDlUrl(url);

      setLastMs(ms);
      setStatus(`Done in ${ms.toFixed(1)} ms (${layoutUsed})`);
    } catch (e) {
      setStatus(`Error: ${getErrorMessage(e)}`);
    } finally {
      setIsRunning(false);
    }
  }, [ready, imgUrl, isRunning, dlUrl, modelKey]);

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
