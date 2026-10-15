/** Cross-window invalidation only. Source code and judging results never leave the API/cache. */
export type PracticeChange = {
  owner: string;
  kind: 'draft' | 'submission';
  problem?: string;
  revision?: string;
  source?: string;
};
type Message = PracticeChange & { nonce: string };
const channelName = 'paperleet.practice.changed.v1';
const eventName = 'paperleet:practice-changed';
function parse(value: unknown): Message | undefined {
  if (!value || typeof value !== 'object') return;
  const data = value as Partial<Message>;
  if (typeof data.owner !== 'string' || !data.owner.startsWith('user:') || data.owner.length > 200
    || !['draft', 'submission'].includes(data.kind || '') || typeof data.nonce !== 'string'
    || data.nonce.length > 200 || (data.problem !== undefined && (typeof data.problem !== 'string' || data.problem.length > 200))
    || (data.source !== undefined && (typeof data.source !== 'string' || data.source.length > 200))
    || (data.revision !== undefined && (typeof data.revision !== 'string' || data.revision.length > 200))) return;
  return {owner: data.owner, kind: data.kind!, nonce: data.nonce,
    ...(data.problem ? {problem: data.problem} : {}), ...(data.revision ? {revision: data.revision} : {}),
    ...(data.source ? {source: data.source} : {})};
}
export function publishPracticeChange(change: PracticeChange) {
  const message = parse({...change, nonce: `${Date.now()}-${Math.random()}`});
  if (!message) return;
  window.dispatchEvent(new CustomEvent(eventName, {detail: message}));
  let channel: BroadcastChannel | undefined;
  try {
    channel = new BroadcastChannel(channelName);
    channel.postMessage(message);
  } catch { /* Storage remains a fallback for older/private browsers. */ }
  finally { try { channel?.close(); } catch { /* Optional transport. */ } }
  try { localStorage.setItem(channelName, JSON.stringify(message)); } catch { /* Focus also revalidates. */ }
}
export function subscribePracticeChanges(listener: (change: PracticeChange) => void) {
  const seen = new Set<string>();
  const receive = (value: unknown) => {
    const message = parse(value);
    if (!message || seen.has(message.nonce)) return;
    seen.add(message.nonce);
    if (seen.size > 100) seen.delete(seen.values().next().value!);
    listener(message);
  };
  const local = (event: Event) => receive((event as CustomEvent).detail);
  const storage = (event: StorageEvent) => {
    if (event.key !== channelName || !event.newValue) return;
    try { receive(JSON.parse(event.newValue)); } catch { /* Ignore unrelated/corrupt invalidations. */ }
  };
  let channel: BroadcastChannel | undefined;
  try { channel = new BroadcastChannel(channelName); channel.onmessage = event => receive(event.data); } catch { /* Optional. */ }
  window.addEventListener(eventName, local);
  window.addEventListener('storage', storage);
  return () => {
    try { channel?.close(); } catch { /* Listener cleanup must still run. */ }
    window.removeEventListener(eventName, local);
    window.removeEventListener('storage', storage);
  };
}
