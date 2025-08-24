"use client";

import OnnxControls from "@/components/OnnXControls";
import { RangeMode, loadImage, rasterize, runAutoLayout } from "@/lib/onnx";
import { createSession } from "@/lib/ort";
import { useEffect, useRef, useState } from "react";

export default function OnnxTesterPage() {
  const [modelUrl, setModelUrl] = useState("/models/animeganv3_arcane.onnx");
  const [imgUrl, setImgUrl] = useState("");
  const [range, setRange] = useState<RangeMode>("0to1");
  const [resizeMax, setResizeMax] = useState(512);

  const [status, setStatus] = useState("Pick a model and an image.");
  const [ready, setReady] = useState(false);
  const [lastMs, setLastMs] = useState<number | null>(null);

  const outCanvasRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);

  const sessionRef = useRef<Awaited<ReturnType<typeof createSession>> | null>(
    null
  );
  const inputNameRef = useRef<string>("");

  // Load model on URL change
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        setReady(false);
        setStatus("Loading model…");
        const s = await createSession(modelUrl);
        if (canceled) return;
        sessionRef.current = s;
        inputNameRef.current = s.inputNames[0];
        setReady(true);
        setStatus(`Model ready: ${s.inputNames[0]} → ${s.outputNames[0]}`);
      } catch (e: any) {
        sessionRef.current = null;
        setStatus(`Model load failed: ${e?.message ?? String(e)}`);
      }
    })();
    return () => {
      canceled = true;
    };
  }, [modelUrl]);

  // Handlers from controls
  const onPickModel = (f: File) => setModelUrl(URL.createObjectURL(f));
  const onPickImage = (f: File) => setImgUrl(URL.createObjectURL(f));

  const onRun = async () => {
    if (!ready || !imgUrl || !sessionRef.current) return;
    try {
      setStatus("Preparing image…");
      const img = await loadImage(imgUrl);
      const { width, height, rgba } = rasterize(
        img,
        scratchRef.current!,
        resizeMax
      );

      setStatus("Running…");
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

      setLastMs(ms);
      setStatus(`Done in ${ms.toFixed(1)} ms (layout ${layoutUsed})`);
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? String(e)}`);
    }
  };

  return (
    <main className="min-h-dvh bg-neutral-900 text-neutral-100">
      <div className="mx-auto w-full max-w-6xl space-y-6 p-6">
        <h1 className="text-2xl font-semibold">
          AnimeGANv3 (ONNX) — Next.js + Tailwind
        </h1>

        <OnnxControls
          ready={ready}
          status={status}
          resizeMax={resizeMax}
          range={range}
          onPickModel={onPickModel}
          onPickImage={onPickImage}
          onChangeResize={setResizeMax}
          onChangeRange={setRange}
          onRun={onRun}
          runDisabled={!ready || !imgUrl}
        />

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
            <h3 className="mb-2 text-sm font-medium text-neutral-200">
              Original
            </h3>
            {imgUrl ? (
              <img
                src={imgUrl}
                alt=""
                className="w-full rounded-lg border border-neutral-800 bg-neutral-900"
              />
            ) : (
              <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-neutral-800 text-neutral-500">
                No image selected
              </div>
            )}
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-4">
            <h3 className="mb-2 text-sm font-medium text-neutral-200">
              Output
            </h3>
            <canvas
              ref={outCanvasRef}
              className="w-full rounded-lg border border-neutral-800 bg-neutral-900"
            />
            {lastMs !== null && (
              <div className="mt-2 text-xs text-neutral-400">
                Inference time: {lastMs.toFixed(1)} ms
              </div>
            )}
          </div>
        </div>

        {/* Hidden scratch canvas for rasterization */}
        <canvas ref={c => (scratchRef.current = c)} className="hidden" />
      </div>
    </main>
  );
}
