import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { useI18n } from "./i18n";
import { AVATAR_MAX_ZOOM, AvatarImageError, avatarCrop, decodeProfileAvatar, encodeProfileAvatar } from "./profileAvatar";
import type { AvatarCropAdjustment, DecodedAvatar } from "./profileAvatar";
import "./avatar-cropper.css";

type Props = { file: File; onConfirm: (image: string) => void; onCancel: () => void; onError: (error: unknown) => void };
const centered = { zoom: 1, centerX: 0.5, centerY: 0.5 };

export default function AvatarCropper({ file, onConfirm, onCancel, onError }: Props) {
  const { t } = useI18n();
  const titleId = useId(), helpId = useId();
  const [image, setImage] = useState<DecodedAvatar>();
  const [adjustment, setAdjustment] = useState(centered);
  const dialog = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const callbacks = useRef({ onCancel, onError });
  callbacks.current = { onCancel, onError };
  const drag = useRef<{ id: number; x: number; y: number; centerX: number; centerY: number; scale: number } | undefined>(undefined);

  useEffect(() => {
    let active = true;
    let decoded: DecodedAvatar | undefined;
    const controller = new AbortController();
    setImage(undefined);
    setAdjustment(centered);
    drag.current = undefined;
    void decodeProfileAvatar(file, controller.signal).then(value => {
      if (!active) { value.release(); return; }
      decoded = value;
      setImage(value);
    }).catch(error => { if (active) callbacks.current.onError(error); });
    return () => { active = false; controller.abort(); decoded?.release(); };
  }, [file]);

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.focus();
    const keyboard = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault(); event.stopPropagation(); callbacks.current.onCancel();
      } else if (event.key === "Tab") {
        const controls = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex="0"]') || [])];
        const first = controls[0], last = controls.at(-1);
        if (!first) { event.preventDefault(); return; }
        if (event.shiftKey && (!dialog.current?.contains(document.activeElement) || document.activeElement === first || document.activeElement === dialog.current)) {
          event.preventDefault(); last?.focus();
        } else if (!event.shiftKey && (!dialog.current?.contains(document.activeElement) || document.activeElement === last || document.activeElement === dialog.current)) {
          event.preventDefault(); first.focus();
        }
      }
    };
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("keydown", keyboard);
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus?.();
    };
  }, []);

  useEffect(() => {
    if (!image || !canvas.current) return;
    try {
      const context = canvas.current.getContext("2d");
      if (!context) throw new AvatarImageError("decode");
      const crop = avatarCrop(image.width, image.height, adjustment);
      context.fillStyle = "#fff";
      context.fillRect(0, 0, 512, 512);
      context.drawImage(image.source, crop.x, crop.y, crop.side, crop.side, 0, 0, 512, 512);
    } catch (error) { callbacks.current.onError(error); }
  }, [image, adjustment]);

  function adjust(next: AvatarCropAdjustment) {
    if (!image) return;
    const crop = avatarCrop(image.width, image.height, next);
    setAdjustment({
      zoom: Math.min(image.width, image.height) / crop.side,
      centerX: (crop.x + crop.side / 2) / image.width,
      centerY: (crop.y + crop.side / 2) / image.height,
    });
  }

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    if (!image || event.button !== 0 || event.isPrimary === false || drag.current) return;
    const width = event.currentTarget.getBoundingClientRect().width;
    if (!width) return;
    event.preventDefault();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, centerX: adjustment.centerX, centerY: adjustment.centerY, scale: avatarCrop(image.width, image.height, adjustment).side / width };
  }

  function moveDrag(event: PointerEvent<HTMLDivElement>) {
    const start = drag.current;
    if (!image || !start || event.pointerId !== start.id) return;
    adjust({ ...adjustment, centerX: start.centerX - (event.clientX - start.x) * start.scale / image.width, centerY: start.centerY - (event.clientY - start.y) * start.scale / image.height });
  }

  function stopDrag(event: PointerEvent<HTMLDivElement>) {
    if (drag.current?.id === event.pointerId) drag.current = undefined;
  }

  function moveWithKeys(event: KeyboardEvent<HTMLDivElement>) {
    const directions: Record<string, [number, number]> = { ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    const direction = directions[event.key];
    if (!image || !direction) return;
    event.preventDefault();
    const step = avatarCrop(image.width, image.height, adjustment).side * (event.shiftKey ? 0.1 : 0.02);
    adjust({ ...adjustment, centerX: adjustment.centerX + direction[0] * step / image.width, centerY: adjustment.centerY + direction[1] * step / image.height });
  }

  function confirm() {
    if (!image) return;
    try { onConfirm(encodeProfileAvatar(image, adjustment)); }
    catch (error) { onError(error); }
  }

  return createPortal(<div className="avatar-cropper-backdrop">
    <div className="avatar-cropper" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={helpId} tabIndex={-1} ref={dialog}>
      <header><h2 id={titleId}>{t("裁剪头像", "Crop profile picture")}</h2><button className="avatar-cropper-close" type="button" aria-label={t("取消裁剪", "Cancel crop")} onClick={onCancel}><X size={20} /></button></header>
      <div className="avatar-cropper-frame" role="group" tabIndex={image ? 0 : -1} aria-label={t("头像裁剪区域，使用方向键移动图片", "Avatar crop area. Use arrow keys to move the image.")} aria-describedby={helpId}
        onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={stopDrag} onPointerCancel={stopDrag} onLostPointerCapture={stopDrag} onKeyDown={moveWithKeys}>
        <canvas ref={canvas} width={512} height={512} aria-hidden="true" />
        {image ? <span className="avatar-cropper-grid" aria-hidden="true" /> : <span className="avatar-cropper-loading" role="status">{t("正在加载图片…", "Loading picture…")}</span>}
      </div>
      <p id={helpId}>{t("拖动或使用方向键调整位置", "Drag or use arrow keys to reposition")}</p>
      <label className="avatar-cropper-zoom"><span>{t("缩放", "Zoom")}</span><input type="range" min="1" max={AVATAR_MAX_ZOOM} step="0.01" value={adjustment.zoom} disabled={!image} onChange={event => { drag.current = undefined; adjust({ ...adjustment, zoom: Number(event.target.value) }); }} /><output>{Math.round(adjustment.zoom * 100)}%</output></label>
      <footer><button type="button" className="button subtle" onClick={onCancel}>{t("取消", "Cancel")}</button><button type="button" className="button primary" disabled={!image} onClick={confirm}>{t("确认裁剪", "Use picture")}</button></footer>
    </div>
  </div>, document.body);
}
