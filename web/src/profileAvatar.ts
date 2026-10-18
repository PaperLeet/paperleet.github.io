export const AVATAR_MAX_FILE_BYTES = 5 * 1024 * 1024;
export const AVATAR_SIZE = 256;
export const AVATAR_MAX_ZOOM = 4;
const imageTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

export type AvatarCropAdjustment = { zoom?: number; centerX?: number; centerY?: number };
export type DecodedAvatar = { source: CanvasImageSource; width: number; height: number; release: () => void };

export {safeAvatarPath} from "./profileAvatarPath";

export class AvatarImageError extends Error {
  constructor(public code: "type" | "size" | "decode") { super(code); }
}

export function avatarCrop(width: number, height: number, adjustment: AvatarCropAdjustment = {}) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) throw new AvatarImageError("decode");
  const finite = (value: number | undefined, fallback: number) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
  const zoom = Math.max(1, Math.min(AVATAR_MAX_ZOOM, finite(adjustment.zoom, 1)));
  const side = Math.min(width, height) / zoom;
  return {
    x: Math.max(0, Math.min(width - side, finite(adjustment.centerX, 0.5) * width - side / 2)),
    y: Math.max(0, Math.min(height - side, finite(adjustment.centerY, 0.5) * height - side / 2)),
    side,
  };
}

export function validateProfileAvatar(file: File) {
  if (!imageTypes.has(file.type)) throw new AvatarImageError("type");
  if (!file.size || file.size > AVATAR_MAX_FILE_BYTES) throw new AvatarImageError("size");
}

/** Call release after using the decoded image, including when an async load is abandoned. */
export async function decodeProfileAvatar(file: File, signal?: AbortSignal): Promise<DecodedAvatar> {
  validateProfileAvatar(file);
  let bitmap: ImageBitmap | undefined;
  let objectUrl: string | undefined;
  const release = () => {
    bitmap?.close();
    bitmap = undefined;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = undefined;
  };
  try {
    if (signal?.aborted) throw new AvatarImageError("decode");
    let source: CanvasImageSource;
    let width: number;
    let height: number;
    if (typeof createImageBitmap === "function") {
      bitmap = await createImageBitmap(file);
      source = bitmap;
      width = bitmap.width;
      height = bitmap.height;
    } else {
      const image = new Image();
      objectUrl = URL.createObjectURL(file);
      image.src = objectUrl;
      let abort: (() => void) | undefined;
      try {
        await Promise.race([
          image.decode(),
          new Promise<never>((_, reject) => {
            abort = () => { release(); image.src = ""; reject(new AvatarImageError("decode")); };
            signal?.addEventListener("abort", abort, { once: true });
          }),
        ]);
      } finally { if (abort) signal?.removeEventListener("abort", abort); }
      source = image;
      width = image.naturalWidth;
      height = image.naturalHeight;
    }
    if (signal?.aborted) throw new AvatarImageError("decode");
    avatarCrop(width, height);
    return { source, width, height, release };
  } catch (error) {
    release();
    if (error instanceof AvatarImageError) throw error;
    throw new AvatarImageError("decode");
  }
}

export function encodeProfileAvatar(image: DecodedAvatar, adjustment?: AvatarCropAdjustment): string {
  try {
    const crop = avatarCrop(image.width, image.height, adjustment);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = AVATAR_SIZE;
    const context = canvas.getContext("2d");
    if (!context) throw new AvatarImageError("decode");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, AVATAR_SIZE, AVATAR_SIZE);
    context.drawImage(image.source, crop.x, crop.y, crop.side, crop.side, 0, 0, AVATAR_SIZE, AVATAR_SIZE);
    const data = canvas.toDataURL("image/jpeg", 0.9);
    if (!/^data:image\/jpeg;base64,[A-Za-z0-9+/]+=*$/.test(data)) throw new AvatarImageError("decode");
    return data;
  } catch (error) {
    if (error instanceof AvatarImageError) throw error;
    throw new AvatarImageError("decode");
  }
}

/** Decode and normalize locally; selecting a file never makes a network request. */
export async function prepareProfileAvatar(file: File, adjustment?: AvatarCropAdjustment): Promise<string> {
  const image = await decodeProfileAvatar(file);
  try { return encodeProfileAvatar(image, adjustment); }
  finally { image.release(); }
}
