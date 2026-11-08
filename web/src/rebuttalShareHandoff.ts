/** Invitation secrets stay in the fragment and this tab's short-lived handoff. */
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem' | 'removeItem'>;
const key = 'paperleet.rebuttal.share-handoff.v1';
const lifetime = 30 * 60 * 1000;
export const rebuttalJoinHref = '#/rebuttal/join';
export const isRebuttalShareToken = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9_-]{43}$/.test(value);

export function makeRebuttalShareToken() {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}
export function rebuttalShareHref(token: string) {
  if (!isRebuttalShareToken(token)) throw Error('invalid');
  return `${rebuttalJoinHref}/${token}`;
}
export function rememberRebuttalShareToken(storage: Storage, token: string, now = Date.now()) {
  if (!isRebuttalShareToken(token)) { storage.removeItem(key); throw Error('invalid'); }
  storage.removeItem(key);
  storage.setItem(key, JSON.stringify({ version: 1, token, createdAt: now }));
}
export function readRebuttalShareToken(storage: Storage, now = Date.now()): string | undefined {
  const raw = storage.getItem(key);
  if (raw === null) return;
  try {
    if (raw.length > 512) throw Error('invalid');
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('invalid');
    const item = value as Record<string, unknown>;
    if (item.version !== 1 || !isRebuttalShareToken(item.token) || typeof item.createdAt !== 'number' || !Number.isFinite(item.createdAt) ||
      item.createdAt > now + 60000 || now - item.createdAt > lifetime) throw Error('expired');
    return item.token;
  } catch (error) { storage.removeItem(key); throw error; }
}
export function forgetRebuttalShareToken(storage: Storage) { storage.removeItem(key); }
