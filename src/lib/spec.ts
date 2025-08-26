import manifest from "@/constants/model-manifest.json" assert { type: "json" };

type Entry = {
  input: string;
  layout: "NCHW" | "NHWC" | "unknown";
  H: number | null;
  W: number | null;
  dtype: string;
  ir_version: number;
  opset: number;
};

export function specFor(publicPath: string): Entry | null {
  // manifest keys are relative to /public/models, e.g. "fns_candy.onnx"
  const key = publicPath.replace(/^\/?models\//, "");
  const entry = (manifest as Record<string, Entry | undefined>)[key];
  return entry ?? null;
}
