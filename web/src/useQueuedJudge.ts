import { useEffect, useRef, useState } from "react";
import { useAccount } from "./social";
import { JobRequestError, requestJob, watchJob } from "./judgeJobs";
import type { JudgeJob } from "./judgeJobs";
import type { Report } from "./types";

type Pending = { id: string; problem: string };
type View = { owner: string; pending?: Pending; job?: JudgeJob; reconnecting: boolean; error: string };
const uuid = () => crypto.randomUUID().replaceAll("-", "");
const keyFor = (owner: string) => `paperleet.judge.pending.v1:${owner}`;
function clientId() {
  try {
    const saved = sessionStorage.getItem("paperleet.judge.client.v1");
    if (saved && /^[a-f0-9]{32}$/.test(saved)) return saved;
    const id = uuid();
    sessionStorage.setItem("paperleet.judge.client.v1", id);
    return id;
  } catch { return uuid(); }
}
function remember(owner: string, value?: Pending) {
  try {
    if (value) sessionStorage.setItem(keyFor(owner), JSON.stringify(value));
    else sessionStorage.removeItem(keyFor(owner));
  } catch { /* Polling works without storage; refresh recovery needs session storage. */ }
}
function recall(owner: string): Pending | undefined {
  try {
    const value = JSON.parse(sessionStorage.getItem(keyFor(owner)) || "null");
    if (value && /^[a-f0-9]{32}$/.test(value.id) && typeof value.problem === "string") return value;
  } catch { /* Ignore invalid or unavailable session storage. */ }
}
const empty = (owner: string): View => ({ owner, reconnecting: false, error: "" });

export function useQueuedJudge(owner: string, ready: boolean, onReport: (report: Report, owner: string) => void) {
  const { health, user } = useAccount();
  const [client] = useState(clientId);
  const [view, setView] = useState<View>(() => empty(owner));
  const controller = useRef<AbortController | undefined>(undefined);
  const callback = useRef(onReport);
  callback.current = onReport;
  const visible = view.owner === owner ? view : empty(owner);

  async function attach(pending: Pending, runOwner: string, signal: AbortSignal) {
    try {
      const job = await watchJob(pending.id, client, signal,
        (job) => {
          if (job.problem !== pending.problem) throw new JobRequestError("The judge returned a different module", 422);
          if (!signal.aborted) setView({ owner: runOwner, pending, job, reconnecting: false, error: "" });
        },
        (connected) => { if (!signal.aborted) setView((previous) => previous.owner === runOwner ? { ...previous, reconnecting: !connected } : previous); });
      if (signal.aborted) return;
      remember(runOwner);
      setView({ owner: runOwner, job, reconnecting: false, error: job.status === "failed" ? job.error || "Judge job failed" : "" });
      if (job.status === "succeeded" && job.report) callback.current(job.report, runOwner);
    } catch (error) {
      if (signal.aborted) return;
      remember(runOwner);
      setView({ owner: runOwner, reconnecting: false, error: error instanceof Error ? error.message : "Judge job unavailable" });
    }
  }

  useEffect(() => {
    controller.current?.abort();
    setView(empty(owner));
    if (!ready) return;
    const pending = recall(owner);
    if (pending) {
      const next = new AbortController();
      controller.current = next;
      setView({ ...empty(owner), pending });
      void attach(pending, owner, next.signal);
    }
    return () => controller.current?.abort();
  }, [owner, ready]);

  async function submit(problem: string, code: string) {
    if (!ready || visible.pending) return;
    const pending = { id: uuid(), problem };
    const next = new AbortController();
    controller.current?.abort();
    controller.current = next;
    remember(owner, pending);
    setView({ ...empty(owner), pending });
    try {
      const job = await requestJob("", client, next.signal, { ...pending, code,
        ...(health?.mode === "hosted" ? { expected_user_id: user?.id } : {}) });
      if (next.signal.aborted) return;
      if (job.id !== pending.id || job.problem !== pending.problem) throw new JobRequestError("The judge returned a different submission", 502);
      setView({ ...empty(owner), pending, job });
      void attach(pending, owner, next.signal);
    } catch (error) {
      if (next.signal.aborted) return;
      if (!(error instanceof JobRequestError) || error.status >= 500) {
        // Creation may have succeeded before the reply was lost. Query its
        // stable ID rather than submit the same source a second time.
        setView({ ...empty(owner), pending, reconnecting: true });
        void attach(pending, owner, next.signal);
      } else {
        remember(owner);
        setView({ ...empty(owner), error: error.message });
      }
    }
  }

  async function cancel() {
    if (!visible.pending || visible.job?.status !== "queued") return;
    try {
      await requestJob(`/${visible.pending.id}/cancel`, client, controller.current!.signal,
        health?.mode === "hosted" ? { expected_user_id: user?.id } : {});
    } catch (error) {
      if (controller.current?.signal.aborted) return;
      setView((previous) => previous.owner === owner ? { ...previous, error: error instanceof Error ? error.message : "Could not cancel this job" } : previous);
    }
  }
  return { ...visible, active: !!visible.pending, submit, cancel };
}
