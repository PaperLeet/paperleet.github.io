export const AVATAR_MAX_FILE_BYTES = 5 * 1024 * 1024;
export const AVATAR_SIZE = 256;
const imageTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

/** Only the API's validated avatar route may be displayed on public pages. */
export function safeAvatarPath(value: unknown): string | undefined {
  return typeof value === "string" && !/[\r\n]/.test(value) && /^\/api\/avatars\/[a-f0-9]{32}\/[a-f0-9]{64}\.jpg$/.test(value) ? value : undefined;
}

export class AvatarImageError extends Error {
  constructor(public code: "type" | "size" | "decode") { super(code); }
}

export function avatarCrop(width: number, height: number) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) throw new AvatarImageError("decode");
  const side = Math.min(width, height);
  return { x: (width - side) / 2, y: (height - side) / 2, side };
}

/** Decode and normalize locally; selecting a file never makes a network request. */
export async function prepareProfileAvatar(file: File): Promise<string> {
  if (!imageTypes.has(file.type)) throw new AvatarImageError("type");
  if (!file.size || file.size > AVATAR_MAX_FILE_BYTES) throw new AvatarImageError("size");
  let bitmap: ImageBitmap | undefined;
  let objectUrl: string | undefined;
  try {
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
      await image.decode();
      source = image;
      width = image.naturalWidth;
      height = image.naturalHeight;
    }
    const crop = avatarCrop(width, height);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = AVATAR_SIZE;
    const context = canvas.getContext("2d");
    if (!context) throw new AvatarImageError("decode");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, AVATAR_SIZE, AVATAR_SIZE);
    context.drawImage(source, crop.x, crop.y, crop.side, crop.side, 0, 0, AVATAR_SIZE, AVATAR_SIZE);
    const data = canvas.toDataURL("image/jpeg", 0.9);
    if (!/^data:image\/jpeg;base64,[A-Za-z0-9+/]+=*$/.test(data)) throw new AvatarImageError("decode");
    return data;
  } catch (error) {
    if (error instanceof AvatarImageError) throw error;
    throw new AvatarImageError("decode");
  } finally {
    bitmap?.close();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}
