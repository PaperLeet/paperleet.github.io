import { isPassed, type Problem, type Report, type Saved } from './types';

export type DraftExpectation = { owner: string; code?: string; updated?: string };
export function draftSnapshotMatches(saved: Saved | undefined, expected: DraftExpectation) {
  return saved?.code === expected.code && (saved?.updated || '') === (expected.updated || '');
}
export function withPracticeReport(saved: Saved, problem: Problem, report: Report): Saved {
  return {...saved, report, acceptedReport: isPassed(problem,report) ? report
    : isPassed(problem,saved.acceptedReport) ? saved.acceptedReport
    : isPassed(problem,saved.report) ? saved.report : undefined};
}
