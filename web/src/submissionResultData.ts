import { parseSolutionHistory, SolutionHistoryIdentityError, type SolutionRun } from './solutionHistoryData';

export { SolutionHistoryIdentityError };
export const validSubmissionId = (value: string): boolean => /^[a-f0-9]{32}$/.test(value);
export const submissionResultTimeoutMs = 12_000;

/** A result URL identifies one immutable attempt, never the latest attempt for a problem. */
export function parseSubmissionResult(value: unknown, userId: string, runId: string): SolutionRun {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid submission result');
  const data = value as Record<string, unknown>;
  if (data.user_id !== userId) throw new SolutionHistoryIdentityError('Submission result owner changed');
  const row = data.run;
  if (!validSubmissionId(runId) || !row || typeof row !== 'object' || Array.isArray(row)) {
    throw new Error('Invalid submission result');
  }
  const run = row as Record<string, unknown>;
  if (run.id !== runId || typeof run.problem !== 'string' || !run.problem.trim()) {
    throw new Error('Submission result does not match this URL');
  }
  const parsed = parseSolutionHistory({ user_id: userId, runs: [run] }, userId, run.problem)[0];
  if (parsed.accepted && (parsed.report.error || parsed.report.mutation || !parsed.report.tests.length ||
      parsed.report.tests.some(test => !test.passed))) throw new Error('Invalid accepted submission');
  return parsed;
}

export type SubmissionResultState =
  | { status: 'loading' }
  | { status: 'ready'; run: SolutionRun }
  | { status: 'missing' | 'auth' | 'identity' | 'error' };

/** Scoped to one account and URL. Disposal blocks late responses even when a transport ignores abort. */
export function loadSubmissionResult(
  userId: string,
  runId: string,
  fetchResult: (path: string, signal: AbortSignal) => Promise<Response>,
  onChange: (state: SubmissionResultState) => void,
) {
  const controller = new AbortController();
  let active = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const publish = (state: SubmissionResultState) => { if (active) onChange(state); };
  publish({ status: 'loading' });
  const done = (async () => {
    if (!validSubmissionId(runId)) { publish({ status: 'missing' }); return; }
    try {
      // Bound both connection and body reads; a stalled response must leave a retry action.
      const deadline = new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error('Submission result timed out'));
        }, submissionResultTimeoutMs);
      });
      const response = await Promise.race([fetchResult(`/api/practice/runs/${runId}`, controller.signal), deadline]);
      if (!active) return;
      if (response.status === 401 || response.status === 403) { publish({ status: 'auth' }); return; }
      if (response.status === 404) { publish({ status: 'missing' }); return; }
      if (!response.ok) throw new Error('Submission result unavailable');
      const run = parseSubmissionResult(await Promise.race([response.json(), deadline]), userId, runId);
      publish({ status: 'ready', run });
    } catch (error) {
      publish({ status: error instanceof SolutionHistoryIdentityError ? 'identity' : 'error' });
    } finally {
      clearTimeout(timer);
    }
  })();
  return { done, dispose() { active = false; clearTimeout(timer); controller.abort(); } };
}

export function submissionFailurePreview(message: string): string {
  const lines = message.trim().split('\n').filter(line => line.trim());
  const tail = lines.slice(-7).join('\n');
  return tail.length > 1200 ? `…${tail.slice(-1200)}` : tail;
}
