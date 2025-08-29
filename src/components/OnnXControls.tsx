"use client";

import type { PresetKey } from "@/constants/presets";
import { useCallback, useRef, useState } from "react";

type Group = {
  title: string;
  items: { key: PresetKey; label: string; hint?: string }[];
};

type Props = {
  ready: boolean;
  status: string;

  modelKey: PresetKey;
  groups: Group[];
  onChangeModel: (key: PresetKey) => void;

  onPickImage: (f: File) => void;
  onRun: () => void;
  runDisabled: boolean;
  isRunning?: boolean;

  strength: number; // 0..100
  onChangeStrength: (v: number) => void;
};

export default function OnnxControls({
  ready,
  status,
  modelKey,
  groups,
  onChangeModel,
  onPickImage,
  onRun,
  runDisabled,
  isRunning = false,
  strength,
  onChangeStrength,
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
    [onPickImage],
  );

  const currentHint = groups
    .flatMap((g) => g.items)
    .find((i) => i.key === modelKey)?.hint;

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body gap-4">
        <h2 className="card-title">Choose style & image</h2>

        {/* grouped preset pills */}
        <div className="space-y-3">
          {groups.map((group) => (
            <div key={group.title}>
              <div className="text-base-content/60 mb-2 text-xs font-semibold uppercase">
                {group.title}
              </div>
              <div className="flex flex-wrap gap-2">
                {group.items.map((p) => (
                  <button
                    key={p.key}
                    onClick={() => onChangeModel(p.key)}
                    className={`btn btn-sm ${modelKey === p.key ? "btn-primary" : "btn-outline"}`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {currentHint && <div className="text-xs opacity-70">{currentHint}</div>}
        <div className="form-control">
          <div className="label">
            <div className="text-base-content/60 mb-2 text-xs font-semibold uppercase">
              Style Strength
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step="25"
            value={strength}
            onChange={(e) => onChangeStrength(parseInt(e.target.value, 10))}
            className="range range-primary"
          />
          <div className="mt-1 flex justify-between px-2.5 text-[.5rem]">
            <span>|</span>
            <span>|</span>
            <span>|</span>
            <span>|</span>
            <span>|</span>
          </div>
          <div className="text-base-content/60 mt-1 flex justify-between text-[.625rem]">
            <span>Original</span>
            <span>25%</span>
            <span>50%</span>
            <span>75%</span>
            <span>100%</span>
          </div>
        </div>

        {/* image dropzone */}
        <div
          onDragOver={(e) => {
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
            <div className="text-base-content/70 text-sm">
              Drag & drop an image here
            </div>
            <div className="divider my-1">or</div>
            <button
              className="btn btn-outline btn-sm rounded-lg"
              onClick={() => fileRef.current?.click()}
            >
              Browse…
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) =>
                e.target.files?.[0] && onPickImage(e.target.files[0])
              }
            />
            <div className="text-base-content/60 text-xs">
              Your image stays in the browser • Auto-resized (AnimeGAN: 512 max,
              FNS: 224×224)
            </div>
          </div>
        </div>

        {/* action */}
        <div className="card-actions items-center justify-between pt-2">
          <span className="text-xs opacity-70">
            {ready ? status : "Loading model…"}
          </span>
          <button
            onClick={onRun}
            disabled={runDisabled}
            className="btn btn-primary rounded-lg"
          >
            {isRunning ? (
              <>
                <span className="loading loading-spinner" /> Stylizing…
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
