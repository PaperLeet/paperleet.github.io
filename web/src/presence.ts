export const presenceStorageKey = "paperleet.presence.visitor";
const visitorIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
let memoryVisitorId: string | undefined;

/** This random browser identifier is only for counting visits, never authentication. */
export function presenceVisitorId(storage?: Pick<Storage, "getItem" | "setItem">): string {
  try {
    const saved = storage?.getItem(presenceStorageKey);
    if (saved && visitorIdPattern.test(saved)) {
      memoryVisitorId = saved;
      return saved;
    }
  } catch {
    // Private browsers may block storage; retain this tab's in-memory identity.
  }
  if (!memoryVisitorId) {
    const bytes = new Uint8Array(16);
    if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(bytes);
    else bytes.forEach((_, index) => { bytes[index] = Math.floor(Math.random() * 256); });
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    memoryVisitorId = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  try {
    storage?.setItem(presenceStorageKey, memoryVisitorId);
  } catch {
    // Counting remains available without persistent storage.
  }
  return memoryVisitorId;
}

type HeartbeatOptions = {
  visitorId: () => string;
  onChange: (online: number | null) => void;
  fetch?: (path: string, options?: RequestInit) => Promise<Response>;
  schedule?: typeof setTimeout;
  cancel?: typeof clearTimeout;
  now?: () => number;
};

/** Refresh the last-hour visitor count hourly, without polling hidden tabs. */
export function createPresenceHeartbeat(options: HeartbeatOptions) {
  const requestFetch = options.fetch ?? fetch;
  const schedule = options.schedule ?? setTimeout;
  const cancel = options.cancel ?? clearTimeout;
  const now = options.now ?? Date.now;
  let active = false;
  let disposed = false;
  let immediate = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let request: AbortController | undefined;
  let nextRequestAt = 0;
  const interval = 3_600_000;
  const retryInterval = 60_000;

  function clearTimer() {
    if (timer !== undefined) cancel(timer);
    timer = undefined;
  }

  function scheduleNext() {
    if (!active || disposed || request) return;
    clearTimer();
    const remaining = nextRequestAt - now();
    if (remaining <= 0) void heartbeat();
    else timer = schedule(() => void heartbeat(), remaining);
  }

  async function heartbeat() {
    if (!active || disposed || request) return;
    clearTimer();
    immediate = false;
    const controller = new AbortController();
    let timedOut = false;
    request = controller;
    const timeout = schedule(() => {
      timedOut = true;
      controller.abort();
      if (active && !disposed) options.onChange(null);
    }, 8000);
    try {
      const response = await requestFetch("/api/presence/heartbeat", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visitor_id: options.visitorId() }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Presence unavailable");
      const data = await response.json();
      if (!Number.isSafeInteger(data.online) || data.online < 0 ||
          data.window_seconds !== 3600 || data.heartbeat_seconds !== 3600) {
        throw new Error("Invalid presence response");
      }
      if (active && !disposed && !controller.signal.aborted) {
        nextRequestAt = now() + interval;
        options.onChange(data.online);
      }
    } catch {
      if (active && !disposed && (!controller.signal.aborted || timedOut)) options.onChange(null);
    } finally {
      cancel(timeout);
      request = undefined;
      if (active && !disposed) {
        if (immediate) nextRequestAt = 0;
        else if (nextRequestAt <= now()) nextRequestAt = now() + retryInterval;
        scheduleNext();
      }
    }
  }

  return {
    resume(force = false) {
      if (disposed || active) return;
      active = true;
      if (force) nextRequestAt = 0;
      if (request) immediate = true;
      else scheduleNext();
    },
    pause() {
      if (disposed) return;
      active = false;
      immediate = false;
      clearTimer();
      request?.abort();
    },
    dispose() {
      disposed = true;
      active = false;
      clearTimer();
      request?.abort();
    },
  };
}
