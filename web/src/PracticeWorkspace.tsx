import { useEffect, useRef, useState } from "react";
import { parseSaved } from "./reports";
import { isPassed } from "./types";
import type { Problem, Report, Saved } from "./types";
import { SocialError, errorMessage, socialApi, useAccount } from "./social";

const LOCAL_KEY = "paperleet.workspace.v1";
type PendingDraft = {
  code: string;
  updated: string;
  base: string;
  revision: number;
};
type Bundle = {
  owner: string;
  records: Record<string, Saved>;
  pending: Record<string, PendingDraft>;
  conflicts: Record<string, Saved>;
  hydrated: boolean;
};
const emptyBundle = (owner: string): Bundle => ({
  owner,
  records: {},
  pending: {},
  conflicts: {},
  hydrated: false,
});
export type DraftExpectation = { owner: string; code?: string; updated?: string };
export function draftSnapshotMatches(saved: Saved | undefined, expected: DraftExpectation) {
  return saved?.code === expected.code && (saved?.updated || '') === (expected.updated || '');
}
function readLocal(): Record<string, Saved> {
  try {
    return parseSaved(JSON.parse(localStorage.getItem(LOCAL_KEY) || "{}"));
  } catch {
    return {};
  }
}
export function practiceOwner(
  mode: "local" | "hosted" | undefined,
  userId?: string,
  accountLoading = false,
) {
  return !mode
    ? "unresolved"
    : mode === "local"
      ? "local"
      : accountLoading
        ? "unresolved"
        : userId
          ? `user:${userId}`
          : "anonymous";
}
export function practiceCacheKey(owner: string, origin: string) {
  return `paperleet.practice.pending.v1:${encodeURIComponent(origin)}:${owner}`;
}
function cacheKey(owner: string) {
  return practiceCacheKey(owner, location.origin);
}
export function mergeWorkspaceSnapshot(
  serverValue: unknown,
  pending: Record<string, PendingDraft>,
) {
  const server = parseSaved(serverValue),
    records = { ...server },
    conflicts: Record<string, Saved> = {},
    dirty = { ...pending };
  for (const [id, draft] of Object.entries(pending)) {
    const saved = server[id] || {};
    if (saved.code === draft.code) {
      delete dirty[id];
      continue;
    }
    records[id] = { ...saved, code: draft.code };
    if ((saved.updated || "") !== draft.base) conflicts[id] = saved;
  }
  return { records, pending: dirty, conflicts };
}
export function withPracticeReport(
  saved: Saved,
  problem: Problem,
  report: Report,
): Saved {
  return {
    ...saved,
    report,
    acceptedReport: isPassed(problem, report)
      ? report
      : isPassed(problem, saved.acceptedReport)
        ? saved.acceptedReport
        : isPassed(problem, saved.report)
          ? saved.report
          : undefined,
  };
}
function readPending(owner: string): Record<string, PendingDraft> {
  try {
    const data = JSON.parse(localStorage.getItem(cacheKey(owner)) || "{}");
    return Object.fromEntries(
      Object.entries(data).filter((entry): entry is [string, PendingDraft] => {
        const v = entry[1] as Partial<PendingDraft> | null;
        return (
          !!v &&
          typeof v.code === "string" &&
          typeof v.base === "string" &&
          typeof v.updated === "string" &&
          typeof v.revision === "number"
        );
      }),
    );
  } catch {
    return {};
  }
}
export function usePracticeWorkspace() {
  const {
    user,
    loading: accountLoading,
    health,
    refresh: refreshAccount,
  } = useAccount();
  const owner = practiceOwner(health?.mode, user?.id, accountLoading);
  const hosted = health?.mode === "hosted";
  const [bundle, setBundle] = useState<Bundle>(() => emptyBundle("unresolved"));
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [saving, setSaving] = useState(false),
    [retryVersion, setRetryVersion] = useState(0),
    [flushVersion, setFlushVersion] = useState(0);
  const currentOwner = useRef(owner);
  currentOwner.current = owner;
  const currentBundle = useRef(bundle);
  currentBundle.current = bundle;
  const epoch = useRef(0),
    flushing = useRef(false),
    failed = useRef(false);
  const flushPromise = useRef<Promise<void> | null>(null);
  const storageError = useRef("");
  function persist(next: Bundle) {
    try {
      if (next.owner === "local")
        localStorage.setItem(LOCAL_KEY, JSON.stringify(next.records));
      else if (next.owner !== "unresolved")
        localStorage.setItem(
          cacheKey(next.owner),
          JSON.stringify(next.pending),
        );
      storageError.current = "";
    } catch {
      storageError.current =
        "Browser storage is full. Download your code to keep a backup.";
    }
  }
  function mutate(change: (previous: Bundle) => Bundle, expectedOwner = owner) {
    if (currentOwner.current !== expectedOwner) return;
    const previous = currentBundle.current;
    if (previous.owner !== expectedOwner) return;
    const next = change(previous);
    if (next === previous) return;
    currentBundle.current = next;
    persist(next);
    setBundle(next);
  }

  useEffect(() => {
    const generation = ++epoch.current;
    failed.current = false;
    flushing.current = false;
    flushPromise.current = null;
    setSaving(false);
    setError("");
    if (owner === "unresolved") {
      setBundle(emptyBundle(owner));
      setLoading(true);
      return;
    }
    if (owner === "local") {
      setBundle({
        ...emptyBundle(owner),
        records: readLocal(),
        hydrated: true,
      });
      setLoading(false);
      return;
    }
    const pending = readPending(owner);
    const records = Object.fromEntries(
      Object.entries(pending).map(([id, draft]) => [
        id,
        { code: draft.code, updated: draft.updated },
      ]),
    );
    setBundle({
      ...emptyBundle(owner),
      records,
      pending,
      hydrated: owner === "anonymous",
    });
    if (owner === "anonymous") {
      setLoading(false);
      return;
    }
    setLoading(true);
    socialApi<{ user_id: string; workspace: Record<string, Saved> }>(
      "practice/workspace",
    )
      .then((data) => {
        if (currentOwner.current !== owner || epoch.current !== generation)
          return;
        if (`user:${data.user_id}` !== owner)
          throw new SocialError(
            "The active account changed. Reloading your account workspace.",
            409,
          );
        const next = {
          owner,
          ...mergeWorkspaceSnapshot(data.workspace, pending),
          hydrated: true,
        };
        setBundle(next);
        currentBundle.current = next;
        persist(next);
      })
      .catch((e) => {
        if (currentOwner.current !== owner || epoch.current !== generation)
          return;
        failed.current = true;
        setError(errorMessage(e));
        if (e instanceof SocialError && e.status === 409) void refreshAccount();
      })
      .finally(() => {
        if (currentOwner.current === owner && epoch.current === generation)
          setLoading(false);
      });
  }, [owner, retryVersion]);
  async function flush(): Promise<void> {
    if (!hosted || !owner.startsWith("user:")) return;
    if (flushPromise.current) {
      await flushPromise.current;
      return;
    }
    if (
      !Object.keys(currentBundle.current.pending).some(
        (id) => !currentBundle.current.conflicts[id],
      )
    )
      return;
    const generation = epoch.current;
    const task = async () => {
      flushing.current = true;
      setSaving(true);
      try {
        while (currentOwner.current === owner && epoch.current === generation) {
          const [entry] = Object.entries(currentBundle.current.pending).filter(
            ([id]) => !currentBundle.current.conflicts[id],
          );
          if (!entry) break;
          const [id, draft] = entry;
          const response = await socialApi<{ draft: Saved }>("practice/draft", {
            problem: id,
            code: draft.code,
            expected_updated: draft.base,
            expected_user_id: user?.id,
          });
          if (currentOwner.current !== owner || epoch.current !== generation)
            return;
          mutate((previous) => {
            const latest = previous.pending[id];
            if (!latest) return previous;
            const pending = { ...previous.pending },
              records = { ...previous.records };
            if (latest.revision === draft.revision) delete pending[id];
            else
              pending[id] = { ...latest, base: response.draft.updated || "" };
            records[id] = { ...records[id], updated: response.draft.updated };
            return { ...previous, pending, records };
          }, owner);
        }
        if (currentOwner.current === owner && epoch.current === generation)
          setError("");
      } catch (e) {
        if (currentOwner.current !== owner || epoch.current !== generation)
          return;
        failed.current = true;
        setError(errorMessage(e));
        if (e instanceof SocialError && e.status === 409)
          setRetryVersion((v) => v + 1);
        throw e;
      } finally {
        if (currentOwner.current === owner && epoch.current === generation) {
          flushing.current = false;
          flushPromise.current = null;
          setSaving(false);
          setFlushVersion((v) => v + 1);
        }
      }
    };
    const promise = task();
    flushPromise.current = promise;
    await promise;
  }
  useEffect(() => {
    if (
      !hosted ||
      !owner.startsWith("user:") ||
      bundle.owner !== owner ||
      !bundle.hydrated ||
      failed.current ||
      flushing.current
    )
      return;
    if (!Object.keys(bundle.pending).some((id) => !bundle.conflicts[id]))
      return;
    const timer = setTimeout(() => {
      void flush().catch(() => {});
    }, 700);
    return () => clearTimeout(timer);
  }, [bundle, owner, hosted, flushVersion]);
  function update(id: string, change: Partial<Saved>, expected?: DraftExpectation) {
    let applied = false;
    mutate((previous) => {
      if (expected && (previous.conflicts[id] || !draftSnapshotMatches(previous.records[id],expected))) return previous;
      applied = true;
      const timestamp = new Date().toISOString();
      const records = {
        ...previous.records,
        [id]: {
          ...previous.records[id],
          ...change,
          ...(owner === "local" ? { updated: timestamp } : {}),
        },
      };
      const pending = { ...previous.pending };
      if (owner !== "local" && typeof change.code === "string") {
        const old = pending[id];
        pending[id] = {
          code: change.code,
          updated: timestamp,
          base: old?.base ?? previous.records[id]?.updated ?? "",
          revision: (old?.revision || 0) + 1,
        };
      }
      return { ...previous, records, pending };
    },expected?.owner || owner);
    if (applied) { failed.current = false; setError(''); }
    return applied && !storageError.current;
  }
  function applyReport(
    problem: Problem,
    report: Report,
    expectedOwner: string,
  ) {
    mutate((previous) => {
      const saved = previous.records[problem.id] || {};
      return {
        ...previous,
        records: {
          ...previous.records,
          [problem.id]: withPracticeReport(saved, problem, report),
        },
      };
    }, expectedOwner);
  }
  function resolveConflict(id: string, keepLocal: boolean) {
    failed.current = false;
    setError("");
    mutate((previous) => {
      const server = previous.conflicts[id];
      if (!server) return previous;
      const pending = { ...previous.pending },
        conflicts = { ...previous.conflicts },
        records = { ...previous.records };
      if (keepLocal && pending[id])
        pending[id] = { ...pending[id], base: server.updated || "" };
      else {
        delete pending[id];
        records[id] = server;
      }
      delete conflicts[id];
      return { ...previous, pending, conflicts, records };
    });
  }
  const visible = bundle.owner === owner ? bundle : emptyBundle(owner);
  return {
    owner,
    saved: visible.records,
    hosted,
    loading: owner === "unresolved" || loading || bundle.owner !== owner,
    ready: visible.hydrated,
    saving,
    pending: Object.keys(visible.pending).length > 0,
    flush,
    error: error || storageError.current,
    conflicts: visible.conflicts,
    update,
    applyReport,
    resolveConflict,
    retry: () => {
      failed.current = false;
      setError("");
      if (!visible.hydrated) setRetryVersion((v) => v + 1);
      else setFlushVersion((v) => v + 1);
    },
    owns: (expected: string) => currentOwner.current === expected,
  };
}
