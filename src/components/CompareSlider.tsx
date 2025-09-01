// src/components/CompareSlider.tsx
"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  before: string;                // e.g. "/img/before.jpg"
  after: string;                 // e.g. "/img/after.png"
  altBefore?: string;
  altAfter?: string;
  className?: string;            // e.g. "w-full aspect-video rounded-box border"
  initial?: number;              // 0..1 (default 0.5)
  fit?: "cover" | "contain";
  showBadges?: boolean;          // Before/After pills
};

export default function CompareSlider({
  before,
  after,
  altBefore = "Before",
  altAfter = "After",
  className = "",
  initial = 0.5,
  fit = "cover",
  showBadges = true,
}: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [t, setT] = useState(clamp01(initial)); // 0..1
  const dragging = useRef(false);

  const setFromX = useCallback((clientX: number) => {
    const el = rootRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setT(clamp01((clientX - r.left) / r.width));
  }, []);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
    dragging.current = true;
    setFromX(e.clientX);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };
  const onMove = (e: PointerEvent) => dragging.current && setFromX(e.clientX);
  const onUp = () => {
    dragging.current = false;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };
  useEffect(() => () => onUp(), []);

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft")  { e.preventDefault(); setT(v => clamp01(v - step)); }
    if (e.key === "ArrowRight") { e.preventDefault(); setT(v => clamp01(v + step)); }
    if (e.key === "Home")       { e.preventDefault(); setT(0); }
    if (e.key === "End")        { e.preventDefault(); setT(1); }
  };

  const pct = t * 100;
  const fitCls = fit === "contain" ? "object-contain" : "object-cover";

  // Clip left side to 'pct%'. Keep overlay full size so image does NOT squash.
  const clip = `inset(0 ${100 - pct}% 0 0)`; // top right bottom left

  return (
    <div
      ref={rootRef}
      className={`relative overflow-hidden select-none touch-none ${className}`}
      role="slider"
      aria-label="Before / After comparison"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      tabIndex={0}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
    >
      {/* Bottom layer: AFTER (full size) */}
      <img
        src={after}
        alt={altAfter}
        className={`absolute inset-0 h-full w-full ${fitCls}`}
        draggable={false}
      />

      {/* Top layer: BEFORE (full size + clip) */}
      <div
        className="absolute inset-0 will-change-[clip-path]"
        style={{ clipPath: clip, WebkitClipPath: clip }}
        aria-hidden
      >
        <img
          src={before}
          alt={altBefore}
          className={`absolute inset-0 h-full w-full ${fitCls}`}
          draggable={false}
        />
      </div>

      {/* Divider */}
      <div className="absolute inset-y-0" style={{ left: `${pct}%`, transform: "translateX(-0.5px)" }}>
        <div className="h-full w-px bg-base-300/70" />
      </div>

      {/* Handle */}
      <button
        type="button"
        aria-label="Drag to compare"
        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-8 w-8 rounded-full bg-base-100/90 shadow ring-1 ring-base-300"
        style={{ left: `${pct}%` }}
        onPointerDown={onPointerDown}
      />

      {showBadges && (
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 px-3 py-3">
          <div className="flex justify-between">
            <span className="badge badge-neutral backdrop-blur">Before</span>
            <span className="badge badge-primary backdrop-blur">After</span>
          </div>
        </div>
      )}
    </div>
  );
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}
