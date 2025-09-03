"use client";
import * as ort from "onnxruntime-web";

// Centralized ORT setup (done once before any session is created)
let initialized = false;

function ortDebugEnabled(): boolean {
  try {
    if (typeof window !== "undefined") {
      const q = new URLSearchParams(window.location.search);
      if (q.get("ort_debug") === "1") return true;
      if (window.localStorage.getItem("ort_debug") === "1") return true;
    }
  } catch {
    // ignore
  }
  return false;
}

export function initOrt() {
  if (initialized) return;
  // Enable deep ORT logs only when explicitly requested
  ort.env.debug = ortDebugEnabled();
  // Use official CDN for WASM binaries; keep SIMD on and single-threaded for broad compatibility
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.simd = true;
  // single thread avoids SharedArrayBuffer/COOP-COEP requirements
  ort.env.wasm.numThreads = 1;
  initialized = true;
}

// Accept URL/string or bytes; normalize to Uint8Array for strict typings
type ModelSource = string | URL | ArrayBuffer | Uint8Array;

export async function createSession(model: ModelSource) {
  initOrt();

  let bytes: Uint8Array;
  if (typeof model === "string" || model instanceof URL) {
    const res = await fetch(model.toString());
    if (!res.ok) {
      throw new Error(`Failed to fetch model: ${res.status} ${res.statusText}`);
    }
    const buf = await res.arrayBuffer();
    bytes = new Uint8Array(buf);
  } else if (model instanceof Uint8Array) {
    bytes = model;
  } else {
    // ArrayBuffer
    bytes = new Uint8Array(model);
  }

  // Prefer WebGPU when available (better op coverage/perf), fallback to WASM.
  // Be resilient: try multiple optimization levels, then fallback to FP32 model if provided INT8 fails.
  const debug = ortDebugEnabled();
  const tryCreate = async (
    providers: ("webgpu" | "wasm")[],
    graphOptimizationLevel: ort.GraphOptimizationLevel,
    stage: string,
  ) => {
    try {
      const sess = await ort.InferenceSession.create(bytes, {
        executionProviders: providers,
        graphOptimizationLevel,
        // default to warnings+errors only; enable verbose if debug flag is set
        logSeverityLevel: debug ? 0 : 2,
        logVerbosityLevel: debug ? 0 : 0,
      });
      (sess as any).__loadMeta = { stage, source: "int8" };
      return sess;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.warn(`[ORT-LOAD] attempt failed stage=${stage}: ${msg}`);
      throw e;
    }
  };

  const attempts: { stage: string; ok: boolean; message?: string }[] = [];

  try {
    return await tryCreate(["webgpu", "wasm"], "all", "webgpu+wasm/all");
  } catch (e1) {
    attempts.push({ stage: "webgpu+wasm/all", ok: false, message: String(e1) });
    try {
      return await tryCreate(["wasm"], "all", "wasm/all");
    } catch (e2) {
      attempts.push({ stage: "wasm/all", ok: false, message: String(e2) });
      try {
        return await tryCreate(["wasm"], "basic", "wasm/basic");
      } catch (e3) {
        attempts.push({ stage: "wasm/basic", ok: false, message: String(e3) });
        try {
          return await tryCreate(["wasm"], "disabled", "wasm/disabled");
        } catch (e4) {
          attempts.push({ stage: "wasm/disabled", ok: false, message: String(e4) });
          // If the source was a URL/string and looks like an INT8 path, attempt FP32 fallback
          if (typeof model === "string" || model instanceof URL) {
            const original = model.toString();
            const isInt8 = /\/models-int8\//.test(original) && /\.int8\.onnx$/.test(original);
            if (isInt8) {
              // Infer family to pick the right non-quantized directory
              const name = original.split("/").pop() || ""; // e.g. agan_ghibli.int8.onnx
              const base = name.replace(/\.int8\.onnx$/, ".onnx");
              const isAgan = /^agan_/.test(base);
              const fp32 = original
                .replace(/\/models-int8\//, isAgan ? "/models-agan/" : "/models/")
                .replace(/\.int8\.onnx$/, ".onnx");
              try {
                console.warn(`INT8 load failed; falling back to FP32: ${fp32}`);
                const res = await fetch(fp32);
                if (!res.ok) throw new Error(`Failed to fetch fallback ${fp32} (${res.status})`);
                const buf = await res.arrayBuffer();
                const fp32bytes = new Uint8Array(buf);
                try {
                  const sess = await ort.InferenceSession.create(fp32bytes, {
                    executionProviders: ["webgpu", "wasm"],
                    graphOptimizationLevel: "all",
                    logSeverityLevel: debug ? 0 : 2,
                    logVerbosityLevel: debug ? 0 : 0,
                  });
                  (sess as any).__loadMeta = { stage: "fp32/webgpu+wasm/all", source: "fp32" };
                  return sess;
                } catch (e6) {
                  attempts.push({ stage: "fp32/webgpu+wasm/all", ok: false, message: String(e6) });
                  const sess2 = await ort.InferenceSession.create(fp32bytes, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "basic",
                    logSeverityLevel: debug ? 0 : 2,
                    logVerbosityLevel: debug ? 0 : 0,
                  });
                  (sess2 as any).__loadMeta = { stage: "fp32/wasm/basic", source: "fp32" };
                  return sess2;
                }
              } catch (e5) {
                const err = e5 as unknown as Error;
                const summary = {
                  model: original,
                  attempts,
                  fallback_error: String(err?.message || err),
                };
                console.error("[ORT-LOAD] failed with fallback", summary);
                throw new Error(`[ORT-LOAD-FAIL] ${JSON.stringify(summary)}`);
              }
            }
          }
          const summary = { model: typeof model === "string" ? model : "bytes", attempts };
          console.error("[ORT-LOAD] failed", summary);
          throw new Error(`[ORT-LOAD-FAIL] ${JSON.stringify(summary)}`);
        }
      }
    }
  }
}

export { ort };
