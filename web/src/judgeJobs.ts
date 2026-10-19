import { parseReport } from "./reports";
import type { Report } from "./types";
import { apiUrl, apiCredentials } from "./apiClient";

export type JudgeJob = {
  id: string;
  problem: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  ahead: number;
  report?: Report;
  error?: string;
};
const states = new Set(["queued", "running", "succeeded", "failed", "cancelled"]);
export {submissionModuleLoadError,pendingSubmissionModuleLoadError,jobErrorText} from './judgeJobMessages';

export class JobRequestError extends Error {
  constructor(message: string, public status: number, public code?: string, public retryAfter?: string) { super(message); }
}

export function parseJob(value: unknown): JudgeJob {
  const job = value as Partial<JudgeJob> | null;
  if (!job || !/^[a-f0-9]{32}$/.test(job.id || "") || typeof job.problem !== "string" ||
      !states.has(job.status || "") || !Number.isSafeInteger(job.ahead) || job.ahead! < 0) {
    throw new JobRequestError("Invalid judge job response", 422);
  }
  let report: Report | undefined;
  try { report = job.report === undefined ? undefined : parseReport(job.report); }
  catch { throw new JobRequestError("Invalid judge job report", 422); }
  if ((job.status === "succeeded" && !report) || (report && report.problem !== job.problem)) {
    throw new JobRequestError("The judge returned a result for a different module", 422);
  }
  return { id: job.id!, problem: job.problem, status: job.status!, ahead: job.ahead!, report,
    error: typeof job.error === "string" ? job.error : undefined };
}

export async function requestJob(path: string, client: string, signal: AbortSignal, body?: object, fetcher: typeof fetch = fetch): Promise<JudgeJob> {
  const response = await fetcher(apiUrl(`/api/judge/jobs${path}`), {
    credentials: apiCredentials,
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json", "X-PaperLeet-Client-ID": client },
    ...(body ? { body: JSON.stringify(body) } : {}),
    cache: "no-store",
    signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
  });
  const value = await response.json().catch(() => null);
  if (!response.ok) {
    const message = typeof value?.error === 'string' ? value.error : 'Judge job unavailable';
    const code = typeof value?.code === 'string' && value.code.length <= 64 ? value.code : undefined;
    const retry = response.headers.get('Retry-After');
    throw new JobRequestError(message, response.status, code, retry && retry.length <= 128 ? retry : undefined);
  }
  return parseJob(value);
}

export function pollDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const aborted = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", aborted); resolve(); }, milliseconds);
    signal.addEventListener("abort", aborted, { once: true });
  });
}

/** Retry reads after a transport outage, never blindly repeat code execution. */
export async function watchJob(id: string, client: string, signal: AbortSignal,
  onUpdate: (job: JudgeJob) => void, onConnection: (connected: boolean) => void,
  fetcher: typeof fetch = fetch, wait = pollDelay): Promise<JudgeJob> {
  let failures = 0;
  while (!signal.aborted) {
    try {
      const job = await requestJob(`/${id}`, client, signal, undefined, fetcher);
      if (job.id !== id) throw new JobRequestError("The judge returned a different job", 422);
      failures = 0;
      onConnection(true);
      onUpdate(job);
      if (job.status !== "queued" && job.status !== "running") return job;
    } catch (error) {
      if (signal.aborted) throw error;
      if (error instanceof JobRequestError && error.status < 500 && error.status !== 429) throw error;
      failures += 1;
      onConnection(false);
    }
    await wait(failures ? Math.min(failures * 2000, 10000) : 1000, signal);
  }
  throw signal.reason;
}
