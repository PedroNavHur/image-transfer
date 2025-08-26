"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { PRESETS, RESIZE_MAX, PresetKey } from "@/constants/presets";
import { createSession } from "@/lib/ort";
import { RangeMode, loadImage, rasterize, runAutoLayout } from "@/lib/onnx";

type StylizerState = {
  modelKey: PresetKey;
  setModelKey: (k: PresetKey) => void;

  imgUrl: string;
  pickImage: (f: File) => void;

  range: RangeMode;
  setRange: (r: RangeMode) => void;

  status: string;
  ready: boolean;
  isRunning: boolean;
  lastMs: number | null;

  dlUrl: string | null;
  run: () => Promise<void>;

  outCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  scratchRef: React.RefObject<HTMLCanvasElement | null>;
};

export function useOnnxStylizer(
  initial: { modelKey?: PresetKey; range?: RangeMode } = {},
): StylizerState {
  const [modelKey, setModelKey] = useState<PresetKey>(
    initial.modelKey ?? "ghibli",
  );
  const [status, setStatus] = useState<string>("Pick a style and an image.");
  const [range, setRange] = useState<RangeMode>(initial.range ?? "0to1");
  const [ready, setReady] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [lastMs, setLastMs] = useState<number | null>(null);

  const [imgUrl, setImgUrl] = useState<string>("");
  const imgUrlPrev = useRef<string | null>(null);

  const [dlUrl, setDlUrl] = useState<string | null>(null);

  // NOTE: include `| null` in the RefObject type to match React's runtime shape
  const outCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);

  const sessionRef = useRef<Awaited<ReturnType<typeof createSession>> | null>(
    null,
  );
  const inputNameRef = useRef<string>("");

  // Revoke blob URL when replaced/unmounted
  useEffect(() => {
    return () => {
      if (dlUrl) URL.revokeObjectURL(dlUrl);
    };
  }, [dlUrl]);

  // Revoke previous image object URL when replaced/unmounted
  useEffect(() => {
    return () => {
      if (imgUrlPrev.current) URL.revokeObjectURL(imgUrlPrev.current);
    };
  }, []);

  // Load model whenever preset changes
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
      } catch (e: unknown) {
        sessionRef.current = null;
        setStatus(`Model load failed: ${getErrorMessage(e)}`);
      } finally {
        setLastMs(null);
        if (dlUrl) {
          URL.revokeObjectURL(dlUrl);
          setDlUrl(null);
        }
      }
    })();

    return () => {
      canceled = true;
    };
  }, [modelKey]); // PRESETS is a module constant

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
      const { width, height, rgba } = rasterize(img, scratch, RESIZE_MAX);

      setStatus("Stylizing…");
      const result = await runAutoLayout(
        sessionRef.current,
        inputNameRef.current,
        rgba,
        width,
        height,
        range,
      );

      const { rgbaOut, W, H, layoutUsed, ms } = result;

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

      // Create a correctly-typed ImageData and copy pixels
      const imgData = ctx.createImageData(W, H);
      imgData.data.set(rgbaOut as unknown as Uint8ClampedArray);
      ctx.putImageData(imgData, 0, 0);

      const url = await canvasToBlobURL(c);
      if (dlUrl) URL.revokeObjectURL(dlUrl);
      setDlUrl(url);

      setLastMs(ms);
      setStatus(`Done in ${ms.toFixed(1)} ms (${layoutUsed})`);
    } catch (e: unknown) {
      setStatus(`Error: ${getErrorMessage(e)}`);
    } finally {
      setIsRunning(false);
    }
  }, [ready, imgUrl, range, isRunning, dlUrl]);

  return {
    modelKey,
    setModelKey,
    imgUrl,
    pickImage,
    range,
    setRange,
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

async function canvasToBlobURL(
  c: HTMLCanvasElement,
  type: string = "image/png",
): Promise<string> {
  const blob = await new Promise<Blob>((resolve, reject) => {
    c.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("toBlob returned null"))),
      type,
    );
  });
  return URL.createObjectURL(blob);
}
