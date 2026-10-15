/** Public counts only; no account IDs, visitor IDs, or private state are shared. */
export const presenceSnapshotKey = 'paperleet.presence.snapshot.v1';
export const presenceSnapshotMaxAge = 3_600_000;
type Snapshot = { online: number; observedAt: number };

export function parsePresenceSnapshot(value: unknown, now = Date.now()): Snapshot | undefined {
  try {
    const parsed = typeof value === 'string' ? JSON.parse(value) : value;
    if (!parsed || typeof parsed !== 'object' || !Number.isSafeInteger(parsed.online) || parsed.online < 0
      || !Number.isSafeInteger(parsed.observedAt) || parsed.observedAt < 0 || parsed.observedAt > now
      || now - parsed.observedAt >= presenceSnapshotMaxAge) return;
    return { online: parsed.online, observedAt: parsed.observedAt };
  } catch { return; }
}

function newest(current: Snapshot | undefined, incoming: Snapshot | undefined) {
  if (!current) return incoming;
  if (!incoming) return current;
  // Deterministically resolve the rare case of two requests starting in one millisecond.
  return incoming.observedAt > current.observedAt
    || (incoming.observedAt === current.observedAt && incoming.online > current.online) ? incoming : current;
}

export function createPresenceSnapshotSync(options: {
  read: () => string | null;
  write: (value: string) => void;
  onChange: (online: number | null) => void;
  now?: () => number;
}) {
  const now = options.now ?? Date.now;
  let current: Snapshot | undefined;
  function stored() {
    try { return parsePresenceSnapshot(options.read(), now()); } catch { return; }
  }
  function show(next: Snapshot | undefined) {
    current = next;
    options.onChange(next?.online ?? null);
  }
  function refresh() {
    show(newest(parsePresenceSnapshot(current, now()), stored()));
  }
  return {
    refresh,
    receive(value: unknown) {
      // Storage messages update display only; never echo them or issue a heartbeat.
      show(newest(parsePresenceSnapshot(current, now()), parsePresenceSnapshot(value, now())));
    },
    publish(online: number, observedAt: number) {
      const previous = newest(parsePresenceSnapshot(current, now()), stored());
      const next = newest(previous, parsePresenceSnapshot({ online, observedAt }, now()));
      show(next);
      if (next && next !== previous) {
        try { options.write(JSON.stringify(next)); } catch { /* Keep this tab usable without storage. */ }
      }
    },
  };
}
