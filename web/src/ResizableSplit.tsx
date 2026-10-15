import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent, ReactNode } from "react";

type ResizableSplitProps = {
  axis: "horizontal" | "vertical";
  storageKey: string;
  /** Preferred share of the first pane, between 0 and 1. */
  initial: number;
  minFirst: number;
  minSecond: number;
  className?: string;
  as?: "main" | "div";
  label: string;
  resetLabel?: string;
  firstId: string;
  secondId: string;
  collapseAt?: number;
  /** An explicit layout action. Ordinary renders and resizing retain the user's preference. */
  layout?: { ratio: number };
  secondCollapsed?: boolean;
  children: [ReactNode, ReactNode];
};

const handleSize = 8;
const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function readRatio(key: string, fallback: number): number {
  try {
    const stored = window.localStorage.getItem(key);
    if (stored === null || stored.trim() === "") return fallback;
    const value = Number(stored);
    return Number.isFinite(value) && value > 0 && value < 1 ? value : fallback;
  } catch {
    return fallback;
  }
}

function saveRatio(key: string, ratio: number) {
  try {
    window.localStorage.setItem(key, String(ratio));
  } catch {
    // Resizing still works when storage is blocked or full.
  }
}

type Drag = {
  pointerId: number;
  coordinate: number;
  ratio: number;
  available: number;
  min: number;
  max: number;
  moved: boolean;
  cursorStyle: HTMLStyleElement;
};

/** Keeps both panes mounted, including while switching to a narrow screen. */
export default function ResizableSplit({
  axis,
  storageKey,
  initial,
  minFirst,
  minSecond,
  className = "",
  as: Container = "div",
  label,
  resetLabel,
  firstId,
  secondId,
  collapseAt,
  layout,
  secondCollapsed = false,
  children,
}: ResizableSplitProps) {
  const defaultRatio = Number.isFinite(initial) ? clamp(initial, 0.01, 0.99) : 0.5;
  const [preferred, setPreferred] = useState(() => readRatio(storageKey, defaultRatio));
  const preferredRef = useRef(preferred);
  const container = useRef<HTMLElement | null>(null);
  const separator = useRef<HTMLDivElement | null>(null);
  const drag = useRef<Drag | null>(null);
  const [available, setAvailable] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [dragging, setDragging] = useState(false);
  const horizontal = axis === "horizontal";

  const updateRatio = useCallback((ratio: number, persist: boolean) => {
    preferredRef.current = ratio;
    setPreferred(ratio);
    if (persist) saveRatio(storageKey, ratio);
  }, [storageKey]);

  const finishDrag = useCallback((updateState = true) => {
    const current = drag.current;
    if (!current) return;
    drag.current = null;
    current.cursorStyle.remove();
    if (separator.current?.hasPointerCapture(current.pointerId)) {
      separator.current.releasePointerCapture(current.pointerId);
    }
    if (current.moved) saveRatio(storageKey, preferredRef.current);
    if (updateState) setDragging(false);
  }, [storageKey]);

  useEffect(() => {
    updateRatio(readRatio(storageKey, defaultRatio), false);
  }, [storageKey, defaultRatio, updateRatio]);

  useEffect(() => {
    if (!layout || !Number.isFinite(layout.ratio)) return;
    finishDrag();
    updateRatio(clamp(layout.ratio, 0.01, 0.99), true);
  }, [layout, finishDrag, updateRatio]);

  useEffect(() => { if (secondCollapsed) finishDrag(); }, [secondCollapsed, finishDrag]);

  useEffect(() => {
    if (!horizontal || collapseAt === undefined) {
      setCollapsed(false);
      return;
    }
    const media = window.matchMedia(`(max-width: ${collapseAt}px)`);
    const update = () => setCollapsed(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [horizontal, collapseAt]);

  useEffect(() => {
    if (collapsed) finishDrag();
    const element = container.current;
    if (!element) return;
    const measure = () => {
      const css = window.getComputedStyle(element);
      const padding = horizontal
        ? parseFloat(css.paddingLeft) + parseFloat(css.paddingRight)
        : parseFloat(css.paddingTop) + parseFloat(css.paddingBottom);
      const dimension = horizontal ? element.clientWidth : element.clientHeight;
      setAvailable(Math.max(0, dimension - padding - handleSize));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [horizontal, collapsed, finishDrag]);

  useEffect(() => {
    const cancel = () => finishDrag();
    window.addEventListener("blur", cancel);
    return () => {
      window.removeEventListener("blur", cancel);
      finishDrag(false);
    };
  }, [finishDrag]);

  const firstMinimum = Math.max(0, minFirst);
  const secondMinimum = Math.max(0, minSecond);
  const minimumTotal = firstMinimum + secondMinimum;
  // If both minima cannot fit, share the available space proportionally.
  const tightRatio = minimumTotal > 0 ? firstMinimum / minimumTotal : 0.5;
  const min = available > 0
    ? available < minimumTotal ? tightRatio : firstMinimum / available
    : 0;
  const max = available > 0
    ? available < minimumTotal ? tightRatio : 1 - secondMinimum / available
    : 1;
  // Window resizes only clamp the display; the user's preferred ratio survives.
  const ratio = clamp(preferred, min, max);
  const percent = Math.round(ratio * 100);

  const reset = () => updateRatio(defaultRatio, true);

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    if (collapsed || secondCollapsed || event.button !== 0 || !event.isPrimary || available <= 0 || min === max) return;
    event.preventDefault();
    finishDrag();
    event.currentTarget.focus({ preventScroll: true });
    event.currentTarget.setPointerCapture(event.pointerId);
    const cursorStyle = document.createElement("style");
    cursorStyle.textContent = `* { cursor: ${horizontal ? "col-resize" : "row-resize"} !important; user-select: none !important; -webkit-user-select: none !important; }`;
    document.head.appendChild(cursorStyle);
    drag.current = {
      pointerId: event.pointerId,
      coordinate: horizontal ? event.clientX : event.clientY,
      ratio,
      available,
      min,
      max,
      moved: false,
      cursorStyle,
    };
    setDragging(true);
  }

  function moveDrag(event: PointerEvent<HTMLDivElement>) {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    event.preventDefault();
    const coordinate = horizontal ? event.clientX : event.clientY;
    const next = clamp(current.ratio + (coordinate - current.coordinate) / current.available, current.min, current.max);
    if (next === preferredRef.current) return;
    current.moved = true;
    updateRatio(next, false);
  }

  function endPointer(event: PointerEvent<HTMLDivElement>) {
    if (drag.current?.pointerId === event.pointerId) finishDrag();
  }

  function handleKey(event: KeyboardEvent<HTMLDivElement>) {
    if (collapsed || secondCollapsed || event.altKey || event.ctrlKey || event.metaKey) return;
    const decrease = horizontal ? "ArrowLeft" : "ArrowUp";
    const increase = horizontal ? "ArrowRight" : "ArrowDown";
    const step = event.shiftKey ? 0.1 : 0.02;
    let next: number;
    if (event.key === decrease) next = clamp(ratio - step, min, max);
    else if (event.key === increase) next = clamp(ratio + step, min, max);
    else if (event.key === "Home") next = min;
    else if (event.key === "End") next = max;
    else if (event.key === "Enter") next = defaultRatio;
    else return;
    event.preventDefault();
    updateRatio(next, true);
  }

  return (
    <Container
      ref={(element) => { container.current = element; }}
      className={`resizable-split ${className}`.trim()}
      data-axis={axis}
      data-collapsed={collapsed ? "true" : undefined}
      data-second-collapsed={secondCollapsed || undefined}
      data-dragging={dragging ? "true" : undefined}
      style={{
        "--split-first": `${ratio}fr`,
        "--split-second": `${1 - ratio}fr`,
        "--split-handle-size": `${handleSize}px`,
      } as CSSProperties}
    >
      {children[0]}
      <div
        ref={separator}
        className="split-handle"
        role="separator"
        tabIndex={collapsed || secondCollapsed ? -1 : 0}
        hidden={collapsed || secondCollapsed}
        aria-hidden={collapsed || secondCollapsed || undefined}
        aria-label={label}
        aria-orientation={horizontal ? "vertical" : "horizontal"}
        aria-controls={`${firstId} ${secondId}`}
        aria-valuemin={Math.round(min * 100)}
        aria-valuemax={Math.round(max * 100)}
        aria-valuenow={percent}
        aria-valuetext={`${percent}% / ${100 - percent}%`}
        title={resetLabel ? `${label} · ${resetLabel}` : label}
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}
        onLostPointerCapture={endPointer}
        onKeyDown={handleKey}
        onDoubleClick={reset}
      >
        <span className="split-handle-grip" aria-hidden="true" />
      </div>
      {children[1]}
    </Container>
  );
}
