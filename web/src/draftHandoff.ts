export const HANDOFF_KEY = 'paperleet.practice.handoff.v1';
export const HANDOFF_TTL = 60 * 60 * 1000;
const MAX_CODE_BYTES = 128 * 1024;
export type DraftHandoff = { version: 1; id: string; problem: string; code: string; created: number; owner?: string };
export type HandoffStorage = Pick<Storage,'getItem'|'setItem'|'removeItem'>;

export function parseHandoff(raw: string | null, now = Date.now()): DraftHandoff | undefined {
  try {
    if (!raw || raw.length > MAX_CODE_BYTES * 6) return;
    const value = JSON.parse(raw);
    if (!value || value.version !== 1 || typeof value.id !== 'string' || !/^[a-zA-Z0-9_-]{8,100}$/.test(value.id) ||
        typeof value.problem !== 'string' || !/^(foundations|rl|architecture|diffusion|finetuning|optimization|representation|world_models|retrieval|self_evolving|network_biology)\/[a-z0-9_-]+$/.test(value.problem) ||
        typeof value.code !== 'string' || new TextEncoder().encode(value.code).length > MAX_CODE_BYTES ||
        !Number.isSafeInteger(value.created) || value.created > now || now - value.created >= HANDOFF_TTL ||
        (value.owner !== undefined && (typeof value.owner !== 'string' || !/^user:[a-zA-Z0-9_-]{1,100}$/.test(value.owner)))) return;
    // Never transport reports, progress, account credentials or arbitrary fields.
    return { version:1,id:value.id,problem:value.problem,code:value.code,created:value.created,...(value.owner ? {owner:value.owner} : {}) };
  } catch { return; }
}

export function saveHandoff(storage: HandoffStorage, value: DraftHandoff) {
  const parsed = parseHandoff(JSON.stringify(value));
  if (!parsed) throw new Error('Invalid draft handoff');
  storage.setItem(HANDOFF_KEY,JSON.stringify(parsed));
}

/** Binding happens before displaying a private import prompt; it never imports. */
export function claimHandoff(storage: HandoffStorage, owner: string): DraftHandoff | undefined {
  const value = parseHandoff(storage.getItem(HANDOFF_KEY));
  if (!value) { storage.removeItem(HANDOFF_KEY); return; }
  if (owner === 'unresolved') return;
  if (owner === 'local' || (value.owner && value.owner !== owner)) { storage.removeItem(HANDOFF_KEY); return; }
  if (owner === 'anonymous') return value;
  if (!owner.startsWith('user:')) return;
  const claimed = { ...value, owner };
  try { saveHandoff(storage,claimed); }
  catch (error) { storage.removeItem(HANDOFF_KEY); throw error; }
  return claimed;
}

export function consumeHandoff(storage: HandoffStorage, id: string) {
  const value = parseHandoff(storage.getItem(HANDOFF_KEY));
  if (value && value.id !== id) return false;
  storage.removeItem(HANDOFF_KEY);
  return true;
}
