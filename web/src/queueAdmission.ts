/** Only an explicit refusal before admission permits another POST. */
import { JobRequestError, pollDelay, requestJob } from './judgeJobs';
import type { JudgeJob } from './judgeJobs';

export type AdmissionWait = { retryAt: number; deadline: number; source: string; attempting?: boolean };
export const ADMISSION_WINDOW = 120_000;
export const MIN_ADMISSION_DELAY = 10_000;
export const ADMISSION_TIMEOUT = 'The queue stayed full. Automatic waiting stopped; your draft is preserved.';

export function admissionDelay(header: string | undefined, now: number): number {
  const value = header?.trim();
  const delay = value && /^\d+$/.test(value) ? Number(value) * 1000
    : value && /[a-z]/i.test(value) ? Date.parse(value) - now : 0;
  return Math.max(MIN_ADMISSION_DELAY, Number.isFinite(delay) ? delay : 0);
}

export async function admitSubmission(submission: { id: string; problem: string; code: string },
  client: string, signal: AbortSignal, expectedUser: string | undefined,
  onAttempt: () => void, onWait: (state: AdmissionWait) => void,
  options: { fetcher?: typeof fetch; wait?: typeof pollDelay; now?: () => number; wall?: () => number } = {}): Promise<JudgeJob> {
  const clock = options.now || (() => performance.now()), wall = options.wall || Date.now, wait = options.wait || pollDelay;
  const cutoff = clock() + ADMISSION_WINDOW, deadline = wall() + ADMISSION_WINDOW;
  // Snapshot once. Edits in the workspace cannot replace a waiting submission.
  const body = { id: submission.id, problem: submission.problem, code: submission.code,
    ...(expectedUser === undefined ? {} : { expected_user_id: expectedUser }) };
  for (;;) {
    signal.throwIfAborted();
    if (clock() >= cutoff) throw new JobRequestError(ADMISSION_TIMEOUT, 429, 'admission_timeout');
    onAttempt();
    try {
      return await requestJob('', client, signal, body, options.fetcher);
    } catch (error) {
      signal.throwIfAborted();
      if (!(error instanceof JobRequestError) || error.status !== 429 || error.code !== 'queue_full') throw error;
      const delay = Math.max(0, Math.min(cutoff - clock(), admissionDelay(error.retryAfter,wall())));
      onWait({ retryAt:wall()+delay, deadline, source:body.code });
      await wait(delay,signal);
    }
  }
}
