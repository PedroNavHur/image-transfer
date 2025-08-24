export const RESIZE_MAX = 512 as const;

/** Preset ONNX models served from /public/models */
export const PRESETS = {
  ghibli: {
    label: "Ghibli",
    file: "/models/agan_ghibli.onnx",
    hint: "Studio look",
  },
  disney: {
    label: "Disney",
    file: "/models/animeganv3_disney.onnx",
    hint: "Clean & bright",
  },
  cyber: {
    label: "Cyberpunk",
    file: "/models/animeganv3_cyber.onnx",
    hint: "Bold edges",
  },
  oilpaint: {
    label: "Oil Paint",
    file: "/models/animeganv3_oil.onnx",
    hint: "Painterly",
  },
} as const;

export type PresetKey = keyof typeof PRESETS;
