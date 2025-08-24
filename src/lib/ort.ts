// Client-side ORT setup + session factory
"use client";
import * as ort from "onnxruntime-web";

let initialized = false;

export function initOrt() {
  if (initialized) return;
  // Use CDN for WASM binaries (no special headers needed)
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  initialized = true;
}

export async function createSession(model: string | Uint8Array) {
  initOrt();
  return await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

export { ort };
