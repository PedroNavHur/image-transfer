"use client";

import { RangeMode } from "@/lib/onnx";
import { useCallback, useRef, useState } from "react";

type Preset = { key: string; label: string; hint?: string };

type Props = {
  ready: boolean;
  status: string;
  range: RangeMode;

  modelKey: string;
  presets: Preset[];
  onChangeModel: (key: string) => void;

  onPickImage: (f: File) => void;
  onChangeRange: (r: RangeMode) => void;
  onRun: () => void;
  runDisabled: boolean;
  isRunning?: boolean;
};

export default function OnnxControls({
  ready,
  status,
  range,
  modelKey,
  presets,
  onChangeModel,
  onPickImage,
  onChangeRange,
  onRun,
  runDisabled,
  isRunning = false,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback(
    (ev: React.DragEvent<HTMLDivElement>) => {
      ev.preventDefault();
      ev.stopPropagation();
      setDragOver(false);
      const f = ev.dataTransfer.files?.[0];
      if (f) onPickImage(f);
    },
    [onPickImage]
  );

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body gap-4">
        <h2 className="card-title">Choose style & image</h2>

        {/* preset pills */}
        <div className="flex flex-wrap gap-2">
          {presets.map(p => (
            <button
              key={p.key}
              onClick={() => onChangeModel(p.key)}
              className={`btn btn-sm ${
                modelKey === p.key ? "btn-primary" : "btn-outline"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
        {presets.find(p => p.key === modelKey)?.hint && (
          <div className="text-xs opacity-70">
            {presets.find(p => p.key === modelKey)?.hint}
          </div>
        )}

        {/* image dropzone */}
        <div
          onDragOver={e => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`rounded-box border-2 border-dashed p-5 transition ${
            dragOver
              ? "border-primary bg-primary/5"
              : "border-base-300 bg-base-200"
          }`}
        >
          <div className="flex flex-col items-center gap-3">
            <div className="avatar placeholder">
              <div className="bg-base-300 text-base-content/70 w-14 rounded-full">
                <span>📷</span>
              </div>
            </div>
            <div className="text-sm text-base-content/70">
              Drag & drop an image here
            </div>
            <div className="divider my-1">or</div>
            <button
              className="btn btn-outline btn-sm"
              onClick={() => fileRef.current?.click()}
            >
              Browse…
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={e =>
                e.target.files?.[0] && onPickImage(e.target.files[0])
              }
            />
            <div className="text-xs text-base-content/60">
              Your image stays in the browser • Auto-resized to 512px max side
            </div>
          </div>
        </div>

        {/* knobs (only input range now) */}
        <div className="grid grid-cols-1">
          <label className="form-control w-full">
            <div className="label">
              <span className="label-text">Input range</span>
            </div>
            <select
              value={range}
              onChange={e => onChangeRange(e.target.value as RangeMode)}
              className="select select-bordered"
            >
              <option value="0to1">[0,1]</option>
              <option value="m1to1">[-1,1]</option>
            </select>
            <div className="label">
              <span className="label-text-alt opacity-70">
                Flip if colors look off
              </span>
            </div>
          </label>
        </div>

        {/* action */}
        <div className="card-actions items-center justify-between pt-2">
          <span className="text-xs opacity-70">
            {ready ? status : "Loading model…"}
          </span>
          <button
            onClick={onRun}
            disabled={runDisabled}
            className="btn btn-primary"
          >
            {isRunning ? (
              <>
                <span className="loading loading-spinner" />
                Stylizing…
              </>
            ) : (
              "Stylize"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
