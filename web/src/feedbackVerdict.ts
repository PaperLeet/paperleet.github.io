import type {Report} from './types';

/** The last raised exception in a pytest message, without ANSI colour or the "E" gutter. */
function raised(message = '') {
  return message.replace(/\u001b\[[0-9;]*m/g, '').split(/\r\n|\r|\n/).map(line => line.replace(/^E\s+/, '').trim())
    .filter(line => /^[\w.]*(?:Error|Exception)\b/.test(line)).at(-1);
}
/** LeetCode-style verdict for the first failing case: anything raised but an assertion is a runtime error. */
export function runtimeFailure(report: Pick<Report, 'tests'>) {
  const first = report.tests.find(test => !test.passed);
  const error = !!first && !first.comparison && raised(first.message);
  return !!error && !/^(?:[\w.]*\.)?AssertionError\b/.test(error);
}
/** One verdict for a saved run, shared by submission history and the result page. */
export function verdictKind(accepted: boolean, report: Pick<Report, 'tests' | 'error'>) {
  return accepted ? 'accepted' : report.error ? 'incomplete' : !report.tests.some(test => !test.passed) ? 'not-accepted' : runtimeFailure(report) ? 'runtime' : 'wrong';
}
/** Runtime in milliseconds: the measured learner phase, else the sum of per-test durations. */
export function runtimeMs(report: Report) {
  const measured = report.browser_run?.measurements?.learner_ms;
  const summed = report.tests.reduce((total, test) => total + (test.duration_s || 0), 0) * 1000;
  return measured ?? (summed > 0 ? summed : undefined);
}
export const formatRuntime = (ms: number) => ms < 1 ? '< 1 ms' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
