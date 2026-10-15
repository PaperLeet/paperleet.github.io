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
export function jobErrorText(message: string, locale: string): string {
  if (locale !== "zh") return message;
  const translations: Record<string, string> = {
    "The judge queue is full. Your draft is preserved; try again shortly.": "判题队列已满，请稍后再提交；你的草稿已保留。",
    "You already have a queued or running submission. Wait for it to finish.": "你已有一个排队或运行中的任务，请等待该任务完成。",
    "This job has already started and cannot be removed from the queue.": "任务已经开始执行，无法取消排队，请等待结果。",
    "This job does not exist or has expired. Your draft is preserved.": "任务不存在或已过期；草稿仍保留，可以重新提交。",
    "The service restarted during this run. Your draft is preserved; submit again.": "服务在判题时重启，本次任务已中断；草稿保留，请重新提交。",
    "The queue wait limit was reached. Your draft is preserved; submit again.": "排队等待已超时；草稿保留，请重新提交。",
    "The judge could not complete this job. Your draft is preserved.": "本次判题未能完成；草稿保留，请稍后重试。",
    "Sign in to view or submit judge jobs.": "请登录后查看或提交判题任务。",
    "Your account changed. Reload before submitting.": "账号已变更，请刷新页面后再提交。",
    "The judge job service is temporarily unavailable. Your draft is preserved.": "判题任务服务暂时不可用；你的草稿已保留。",
    "The queue stayed full. Automatic waiting stopped; your draft is preserved.": "队列持续繁忙，已停止自动等待；草稿仍保留，可稍后重新提交。",
    "The submission is being confirmed. Wait for its queue status before cancelling.": "正在确认这次提交，确认入队后才能取消排队。",
    "Server submission could not start. Your draft is preserved; try again.": "服务端提交未能启动；草稿已保留，请重试。",
    "Too many requests. Please wait before submitting again.": "提交过于频繁，请稍候再试。",
  };
  return translations[message] || message;
}
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
