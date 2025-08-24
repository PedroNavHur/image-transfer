"use client";

import OnnxControls from "@/components/OnnXControls";
import { PRESETS, PresetKey } from "@/constants/presets";
import { useOnnxStylizer } from "@/hooks/useStylizer";

export default function OnnxTesterPage() {
  const {
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
  } = useOnnxStylizer({ modelKey: "ghibli", range: "0to1" });

  return (
    <main className="min-h-dvh bg-base-200 relative overflow-hidden">
      {/* soft glow */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-32 -left-32 h-80 w-80 rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-40 -right-24 h-96 w-96 rounded-full bg-secondary/20 blur-3xl" />
      </div>

      <div className="mx-auto w-full max-w-7xl p-6 space-y-6">
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
              range={range}
              modelKey={modelKey}
              presets={Object.entries(PRESETS).map(([key, v]) => ({
                key: key as PresetKey,
                label: v.label,
                hint: v.hint,
              }))}
              onChangeModel={k => setModelKey(k as PresetKey)}
              onPickImage={pickImage}
              onChangeRange={setRange}
              onRun={run}
              runDisabled={!ready || !imgUrl || isRunning}
              isRunning={isRunning}
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
                  <div className="rounded-box border border-base-300 bg-base-200 p-2">
                    {imgUrl ? (
                      <img
                        src={imgUrl}
                        alt="Original"
                        className="mx-auto max-h-[60vh] w-full object-contain"
                        draggable={false}
                      />
                    ) : (
                      <div className="grid h-[40vh] place-items-center text-base-content/60 text-sm">
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
                        href={dlUrl ?? "#"}
                        download={`stylized_${modelKey}.png`}
                        aria-disabled={!dlUrl}
                        className={`btn btn-sm btn-secondary ${dlUrl ? "" : "btn-disabled"}`}
                      >
                        Download PNG
                      </a>
                    </div>
                  </div>
                  <div className="rounded-box border border-base-300 bg-base-200 p-2">
                    {dlUrl ? (
                      <img
                        src={dlUrl}
                        alt="Stylized"
                        className="mx-auto max-h-[60vh] w-full object-contain"
                        draggable={false}
                      />
                    ) : (
                      <div className="grid h-[40vh] place-items-center text-base-content/60 text-sm">
                        Run to see result
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* status */}
            <div className="mt-6 alert bg-base-100 border border-base-300">
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
