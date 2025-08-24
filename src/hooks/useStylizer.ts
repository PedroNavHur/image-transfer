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

  outCanvasRef: React.RefObject<HTMLCanvasElement>;
  scratchRef: React.RefObject<HTMLCanvasElement>;
};

export function useOnnxStylizer(
  initial: { modelKey?: PresetKey; range?: RangeMode } = {}
): StylizerState {
  const [modelKey, setModelKey] = useState<PresetKey>(
    initial.modelKey ?? "ghibli"
  );
  const [status, setStatus] = useState("Pick a style and an image.");
  const [range, setRange] = useState<RangeMode>(initial.range ?? "0to1");
  const [ready, setReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [lastMs, setLastMs] = useState<number | null>(null);

  const [imgUrl, setImgUrl] = useState("");
  const imgUrlPrev = useRef<string | null>(null);

  const [dlUrl, setDlUrl] = useState<string | null>(null);

  const outCanvasRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef<HTMLCanvasElement>(null);

  const sessionRef = useRef<Awaited<ReturnType<typeof createSession>> | null>(
    null
  );
  const inputNameRef = useRef<string>("");

  // Revoke blob URL when replaced/unmounted
  useEffect(
    () => () => {
      if (dlUrl) URL.revokeObjectURL(dlUrl);
    },
    [dlUrl]
  );

  // Revoke previous image object URL when replaced
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
        inputNameRef.current = s.inputNames[0];
        setReady(true);
        setStatus(`Ready: ${preset.label}`);
      } catch (e: any) {
        sessionRef.current = null;
        setStatus(`Model load failed: ${e?.message ?? String(e)}`);
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
  }, [modelKey]); // eslint-disable-line react-hooks/exhaustive-deps

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
    [dlUrl]
  );

  const run = useCallback(async () => {
    if (!ready || !imgUrl || !sessionRef.current || isRunning) return;
    try {
      setIsRunning(true);
      setStatus("Preparing image…");

      const img = await loadImage(imgUrl);
      const { width, height, rgba } = rasterize(
        img,
        scratchRef.current!,
        RESIZE_MAX
      );

      setStatus("Stylizing…");
      const { rgbaOut, W, H, layoutUsed, ms } = await runAutoLayout(
        sessionRef.current,
        inputNameRef.current,
        rgba,
        width,
        height,
        range
      );

      const c = outCanvasRef.current!;
      c.width = W;
      c.height = H;
      c.getContext("2d")!.putImageData(new ImageData(rgbaOut, W, H), 0, 0);

      const url = await canvasToBlobURL(c);
      if (dlUrl) URL.revokeObjectURL(dlUrl);
      setDlUrl(url);

      setLastMs(ms);
      setStatus(`Done in ${ms.toFixed(1)} ms (${layoutUsed})`);
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
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

async function canvasToBlobURL(c: HTMLCanvasElement, type = "image/png") {
  const blob = await new Promise<Blob>((resolve, reject) => {
    c.toBlob(
      b => (b ? resolve(b) : reject(new Error("toBlob returned null"))),
      type
    );
  });
  return URL.createObjectURL(blob);
}
