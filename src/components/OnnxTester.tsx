"use client";

import OnnxControls from "@/components/OnnXControls";
import { PRESETS, PRESET_GROUPS } from "@/constants/presets";
import { useOnnxStylizer } from "@/hooks/useStylizer";
import Image from "next/image";

const groups = PRESET_GROUPS.map((g) => ({
  title: g.title,
  items: g.keys.map((k) => ({
    key: k,
    label: PRESETS[k].label,
    hint: PRESETS[k].hint,
  })),
}));

export default function OnnxTester() {
  const {
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
  } = useOnnxStylizer({ modelKey: "ghibli" });

  return (
    <main className="bg-base-200 relative min-h-dvh overflow-hidden">
      {/* soft glow */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="bg-primary/20 absolute -top-32 -left-32 h-80 w-80 rounded-full blur-3xl" />
        <div className="bg-secondary/20 absolute -right-24 -bottom-40 h-96 w-96 rounded-full blur-3xl" />
      </div>

      <div className="mx-auto w-full max-w-7xl space-y-6 p-6">
        {/* Navbar */}
        <div className="navbar rounded-box bg-base-100/80 shadow backdrop-blur">
          <div className="flex-1">
            <span className="btn btn-ghost text-xl">StyleForge</span>
          </div>
          <div className="flex-none items-center gap-2">
            {isRunning && (
              <span className="loading loading-spinner loading-sm text-primary" />
            )}
            <span className={`badge ${ready ? "badge-success" : "badge-info"}`}>
              {ready ? "Ready" : "Loading"}
            </span>
          </div>
        </div>

        {/* Layout: left controls, right gallery */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* LEFT: controls */}
          <aside className="lg:col-span-4">
            <OnnxControls
              ready={ready}
              status={status}
              modelKey={modelKey}
              groups={groups}
              onChangeModel={(k) => setModelKey(k)}
              onPickImage={pickImage}
              onRun={run}
              runDisabled={!ready || !imgUrl || isRunning}
              isRunning={isRunning}
              strength={strength}
              onChangeStrength={setStrength}
            />
          </aside>

          {/* RIGHT: before / after */}
          <section className="lg:col-span-8">
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* BEFORE */}
              <div className="card bg-base-100 shadow-xl">
                <div className="card-body gap-3">
                  <div className="flex items-center justify-between">
                    <h3 className="card-title text-base">Before</h3>
                  </div>
                  <div className="rounded-box border-base-300 bg-base-200 border p-2">
                    {imgUrl ? (
                      <Image
                        src={imgUrl}
                        alt="Original"
                        width={512}
                        height={512}
                        className="mx-auto max-h-[60vh] w-full object-contain"
                        draggable={false}
                      />
                    ) : (
                      <div className="text-base-content/60 grid h-[40vh] place-items-center text-sm">
                        Upload an image to preview
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* AFTER */}
              <div className="card bg-base-100 shadow-xl">
                <div className="card-body gap-3">
                  <div className="flex items-center justify-between">
                    <h3 className="card-title text-base">After</h3>
                    <div className="flex items-center gap-2">
                      {lastMs !== null && (
                        <div className="badge badge-ghost">
                          {lastMs.toFixed(1)} ms
                        </div>
                      )}
                      <a
                        role="button"
                        href={dlUrl ?? "#"}
                        download={`stylized_${modelKey}.png`}
                        aria-disabled={!dlUrl}
                        className={`btn btn-sm btn-secondary rounded-lg ${
                          dlUrl ? "" : "btn-disabled"
                        }`}
                      >
                        Download PNG
                      </a>
                    </div>
                  </div>
                  <div className="rounded-box border-base-300 bg-base-200 border p-2">
                    {dlUrl ? (
                      <Image
                        src={dlUrl}
                        alt="Stylized"
                        width={512}
                        height={512}
                        className="mx-auto max-h-[60vh] w-full object-contain"
                        draggable={false}
                      />
                    ) : (
                      <div className="text-base-content/60 grid h-[40vh] place-items-center text-sm">
                        Run to see result
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* status */}
            <div className="alert bg-base-100 border-base-300 mt-6 border">
              {isRunning ? (
                <span className="flex items-center gap-2 text-sm">
                  <span className="loading loading-spinner loading-xs" />
                  {status}
                </span>
              ) : (
                <span className="text-sm">{status}</span>
              )}
            </div>
          </section>
        </div>

        {/* Hidden canvases */}
        <canvas ref={scratchRef} className="hidden" />
        <canvas ref={outCanvasRef} className="hidden" />
      </div>
    </main>
  );
}
