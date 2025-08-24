"use client";
import * as ort from "onnxruntime-web";

// one-time ORT env setup
let initialized = false;
export function initOrt() {
  if (initialized) return;
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
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

  return await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

export { ort };
