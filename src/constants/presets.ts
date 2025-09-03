export const RESIZE_MAX = 512 as const;

export type Family = "agan" | "fns";

/** Preset ONNX models served from /public/models */
export const PRESETS = {
  // Studio Look (AnimeGANv3; dynamic size)
  ghibli: {
    label: "Ghibli",
    file: "/models-int8/agan_ghibli.int8.onnx",
    hint: "Studio look",
    family: "agan" as const,
  },
  hayao: {
    label: "Hayao",
    file: "/models-int8/agan_hayao.int8.onnx",
    hint: "Studio look",
    family: "agan" as const,
  },
  shinkai: {
    label: "Shinkai",
    file: "/models-int8/agan_shinkai.int8.onnx",
    hint: "Studio look",
    family: "agan" as const,
  },
  sketch: {
    label: "Sketch",
    file: "/models-int8/agan_sketch.int8.onnx",
    hint: "Sketch lines",
    family: "agan" as const,
  },

  // Filter Style (Fast Neural Style; fixed 224×224, NCHW)
  mosaic: {
    label: "Mosaic",
    file: "/models-int8/fns_mosaic.int8.onnx",
    hint: "224×224",
    family: "fns" as const,
  },
  candy: {
    label: "Candy",
    file: "/models-int8/fns_candy.int8.onnx",
    hint: "224×224",
    family: "fns" as const,
  },
  udnie: {
    label: "Udnie",
    file: "/models-int8/fns_udnie.int8.onnx",
    hint: "224×224",
    family: "fns" as const,
  },
  pointilism: {
    label: "Pointilism",
    file: "/models-int8/fns_pointilism.int8.onnx",
    hint: "224×224",
    family: "fns" as const,
  },
} as const;

export type PresetKey = keyof typeof PRESETS;

/** Grouped UI helper */
export const PRESET_GROUPS = [
  {
    title: "Filter Style",
    keys: ["mosaic", "candy", "udnie", "pointilism"] as const,
  },
  {
    title: "Studio Look",
    keys: ["ghibli", "hayao", "shinkai", "sketch"] as const,
  },
] satisfies readonly { title: string; keys: readonly PresetKey[] }[];

/** Small helper for logic */
export function isFNS(key: PresetKey) {
  return PRESETS[key].family === "fns";
}
