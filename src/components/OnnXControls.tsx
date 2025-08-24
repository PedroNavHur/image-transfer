"use client";

import { RangeMode } from "@/src/lib/onnx";

type Props = {
  ready: boolean;
  status: string;
  resizeMax: number;
  range: RangeMode;
  onPickModel: (f: File) => void;
  onPickImage: (f: File) => void;
  onChangeResize: (n: number) => void;
  onChangeRange: (r: RangeMode) => void;
  onRun: () => void;
  runDisabled: boolean;
};

export default function OnnxControls({
  ready,
  status,
  resizeMax,
  range,
  onPickModel,
  onPickImage,
  onChangeResize,
  onChangeRange,
  onRun,
  runDisabled,
}: Props) {
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-xl border border-neutral-800 bg-neutral-950 p-4">
      <div className="space-y-1">
        <div className="text-xs text-neutral-400">Model (.onnx)</div>
        <input
          type="file"
          accept=".onnx"
          onChange={e => e.target.files?.[0] && onPickModel(e.target.files[0])}
          className="block w-64 text-sm file:mr-3 file:rounded-md file:border file:border-neutral-700 file:bg-neutral-800 file:px-3 file:py-1.5 file:text-neutral-100"
        />
      </div>

      <div className="space-y-1">
        <div className="text-xs text-neutral-400">Image</div>
        <input
          type="file"
          accept="image/*"
          onChange={e => e.target.files?.[0] && onPickImage(e.target.files[0])}
          className="block w-64 text-sm file:mr-3 file:rounded-md file:border file:border-neutral-700 file:bg-neutral-800 file:px-3 file:py-1.5 file:text-neutral-100"
        />
      </div>

      <div className="space-y-1">
        <div className="text-xs text-neutral-400">Resize max side (px)</div>
        <input
          type="number"
          min={0}
          value={resizeMax}
          onChange={e => onChangeResize(parseInt(e.target.value || "0", 10))}
          className="w-28 rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm"
        />
      </div>

      <div className="space-y-1">
        <div className="text-xs text-neutral-400">Input range</div>
        <select
          value={range}
          onChange={e => onChangeRange(e.target.value as RangeMode)}
          className="rounded-md border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm"
        >
          <option value="0to1">[0,1]</option>
          <option value="m1to1">[-1,1]</option>
        </select>
      </div>

      <button
        onClick={onRun}
        disabled={runDisabled}
        className="rounded-md border border-neutral-700 bg-neutral-800 px-3 py-1.5 text-sm font-medium hover:bg-neutral-750 disabled:opacity-50"
      >
        Run
      </button>

      <span className="text-xs text-neutral-400">
        {ready ? status : "Loading model…"}
      </span>
    </div>
  );
}
