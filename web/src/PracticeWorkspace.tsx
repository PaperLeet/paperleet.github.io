import { useEffect, useRef, useState } from "react";
import { draftSnapshotMatches, withPracticeReport, type DraftExpectation } from "./practiceRecords";
import { useGuestWorkspace } from "./useGuestWorkspace";
export { draftSnapshotMatches, withPracticeReport, type DraftExpectation } from "./practiceRecords";
import type { Problem, Report, Saved } from "./types";
import { SocialError, errorMessage, socialApi, useAccount } from "./social";
import type { PendingDraft, createHostedDraftCache } from "./hostedDraftCache";
import { publishPracticeChange, subscribePracticeChanges } from "./practiceChanges";

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
export function usePracticeWorkspace() {
  const {
    user,
    loading: accountLoading,
    health,
    refresh: refreshAccount,
  } = useAccount();
  const owner = practiceOwner(health?.mode, user?.id, accountLoading);
  const hosted = health?.mode === "hosted";
  const browserWorkspace = useGuestWorkspace(owner === "local" || owner === "anonymous" ? owner : undefined);
  const [bundle, setBundle] = useState<Bundle>(() => emptyBundle("unresolved"));
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [saving, setSaving] = useState(false),
    [retryVersion, setRetryVersion] = useState(0),
    [flushVersion, setFlushVersion] = useState(0);
  const source = useRef(`workspace-${Date.now()}-${Math.random()}`);
  const cache = useRef<ReturnType<typeof createHostedDraftCache> | null>(null);
  const refreshWorkspace = useRef<(() => void) | null>(null);
  const refreshQueued = useRef(false);
  const serverMutation = useRef(0);
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
      if (next.owner.startsWith("user:"))
        cache.current?.write(next.owner, next.pending);
      storageError.current = "";
    } catch {
      storageError.current =
        "Browser storage is full. Keep this page open and retry syncing your draft.";
    }
  }
  function mutate(change: (previous: Bundle) => Bundle, expectedOwner = owner) {
    if (currentOwner.current !== expectedOwner) return false;
    const previous = currentBundle.current;
    if (previous.owner !== expectedOwner || !cache.current) return false;
    const next = change(previous);
    if (next === previous) return false;
    currentBundle.current = next;
    persist(next);
    setBundle(next);
    return true;
  }

  useEffect(() => () => { currentOwner.current = 'unmounted'; epoch.current += 1; }, []);
  useEffect(() => {
    const generation = ++epoch.current;
    currentOwner.current = owner;
    let disposed = false, initialized = false, requestRunning = false, repeat = false;
    let mergeSnapshot: typeof import("./hostedDraftCache").mergeWorkspaceSnapshot;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let lastRequest = 0;
    const active = () => !disposed && currentOwner.current === owner && epoch.current === generation;
    failed.current = false;
    flushing.current = false;
    flushPromise.current = null;
    refreshQueued.current = false;
    setSaving(false);
    setError("");
    const initial = emptyBundle(owner);
    currentBundle.current = initial;
    setBundle(initial);
    setLoading(owner !== 'local' && owner !== 'anonymous');
    if (!owner.startsWith('user:')) return () => { disposed = true; };

    async function refresh() {
      if (!active() || !initialized) return;
      if (flushing.current) { refreshQueued.current = true; return; }
      if (requestRunning) { repeat = true; return; }
      requestRunning = true;
      lastRequest = Date.now();
      const mutation = serverMutation.current;
      try {
        const data = await socialApi<{user_id: string; workspace: Record<string, Saved>}>('practice/workspace');
        if (!active()) return;
        if (`user:${data.user_id}` !== owner) throw new SocialError('The active account changed. Reloading your account workspace.', 409);
        // A response started before our own commit cannot replace that newer result.
        if (mutation !== serverMutation.current) { repeat = true; return; }
        if (flushing.current) { refreshQueued.current = true; return; }
        const latest = currentBundle.current;
        const next = {owner, ...mergeSnapshot(data.workspace, latest.pending), hydrated: true};
        cache.current?.acknowledge(owner, data.workspace);
        currentBundle.current = next;
        setBundle(next);
        persist(next);
        failed.current = false;
        setError('');
      } catch (e) {
        if (!active()) return;
        if (!currentBundle.current.hydrated) failed.current = true;
        setError(errorMessage(e));
        if (e instanceof SocialError && e.status === 409) void refreshAccount();
      } finally {
        requestRunning = false;
        if (active()) {
          setLoading(false);
          if (repeat) { repeat = false; schedule(); }
        }
      }
    }
    function schedule() {
      if (!active() || timer !== undefined) return;
      timer = setTimeout(() => { timer = undefined; void refresh(); }, 120);
    }
    const becameActive = () => {
      if (document.visibilityState === 'hidden' || Date.now() - lastRequest < 2000) return;
      schedule();
    };
    refreshWorkspace.current = schedule;
    const unsubscribe = subscribePracticeChanges(change => {
      if (change.owner === owner && change.source !== source.current) {
        serverMutation.current += 1;
        schedule();
      }
    });
    window.addEventListener('focus', becameActive);
    window.addEventListener('online', schedule);
    document.addEventListener('visibilitychange', becameActive);
    // Hosted recovery is unnecessary for the guest/local library. Keep the
    // editor disabled until both this optional runtime and the API are ready.
    void import('./hostedDraftCache').then(runtime => {
      if (!active()) return;
      if (!cache.current) cache.current = runtime.createHostedDraftCache(source.current);
      mergeSnapshot = runtime.mergeWorkspaceSnapshot;
      const pending = cache.current.read(owner);
      const restored = {
        ...emptyBundle(owner), pending,
        records: Object.fromEntries(Object.entries(pending).map(([id, draft]) => [id, {code: draft.code, updated: draft.updated}])),
      };
      currentBundle.current = restored;
      setBundle(restored);
      initialized = true;
      void refresh();
    }).catch(e => {
      if (!active()) return;
      failed.current = true;
      setError(errorMessage(e));
      setLoading(false);
    });
    return () => {
      disposed = true;
      clearTimeout(timer);
      unsubscribe();
      window.removeEventListener('focus', becameActive);
      window.removeEventListener('online', schedule);
      document.removeEventListener('visibilitychange', becameActive);
      if (refreshWorkspace.current === schedule) refreshWorkspace.current = null;
    };
  }, [owner, retryVersion]);
  async function flush(): Promise<void> {
    if (!hosted || !owner.startsWith("user:") || currentOwner.current !== owner || currentBundle.current.owner !== owner) return;
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
          serverMutation.current += 1;
          cache.current?.acknowledge(owner, {[id]: {code: draft.code}});
          publishPracticeChange({owner, problem: id, kind: 'draft', revision: response.draft.updated, source: source.current});
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
          if (refreshQueued.current) { refreshQueued.current = false; refreshWorkspace.current?.(); }
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
    if (owner === "local" || owner === "anonymous") return browserWorkspace.update(id,change,expected);
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
    if (owner === "local" || owner === "anonymous") return browserWorkspace.applyReport(problem,report,expectedOwner);
    const applied = mutate((previous) => {
      const saved = previous.records[problem.id] || {};
      return {
        ...previous,
        records: {
          ...previous.records,
          [problem.id]: withPracticeReport(saved, problem, report),
        },
      };
    }, expectedOwner);
    if (applied) {
      serverMutation.current += 1;
      publishPracticeChange({owner: expectedOwner, problem: problem.id, kind: 'submission', source: source.current});
    }
  }
  function resolveConflict(id: string, keepLocal: boolean) {
    if (owner === "local" || owner === "anonymous") return browserWorkspace.resolveConflict(id,keepLocal);
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
  function replaceCode(problem: Problem, code: string, expected: DraftExpectation) {
    if (owner === 'local' || owner === 'anonymous') return browserWorkspace.replaceCode(problem,code,expected);
    const latest = currentBundle.current;
    if (!latest.hydrated || latest.owner !== owner || currentOwner.current !== expected.owner || saving || flushing.current || failed.current
      || Object.keys(latest.pending).length || error || storageError.current || latest.conflicts[problem.id]
      || (latest.records[problem.id]?.code ?? problem.starter) === code) return false;
    return update(problem.id,{code},expected);
  }
  function resetToStarter(problem: Problem, expected: DraftExpectation) {
    return replaceCode(problem,problem.starter,expected);
  }
  if (owner === "local" || owner === "anonymous") return {
    ...browserWorkspace, owner, hosted, flush, update, applyReport, replaceCode, resetToStarter, resolveConflict,
    owns: (expected: string) => currentOwner.current === expected,
  };
  return {
    backups: [] as typeof browserWorkspace.backups,
    restoreBackup: browserWorkspace.restoreBackup,
    discardBackup: browserWorkspace.discardBackup,
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
    replaceCode,
    resetToStarter,
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
