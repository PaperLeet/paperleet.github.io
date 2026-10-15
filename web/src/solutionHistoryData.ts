import { parseReport } from "./reports";
import type { Report } from "./types";

export type SolutionRun = {
  id: string;
  problem: string;
  created_at: string;
  accepted: boolean;
  source_sha256: string;
  code: string;
  report: Report;
};

export type SolutionHistoryState = {
  runs: SolutionRun[];
  offset: number;
  hasMore: boolean;
  loading: boolean;
  error: boolean;
};

export const solutionHistoryPageSize = 30;
export const emptySolutionHistory = (): SolutionHistoryState => ({
  runs: [], offset: 0, hasMore: true, loading: false, error: false,
});

function isObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

export class SolutionHistoryIdentityError extends Error {}

/** Never show a response from a session that changed while the request was in flight. */
export function parseSolutionHistory(value: unknown, userId: string, problem: string): SolutionRun[] {
  if (!isObject(value) || value.user_id !== userId) {
    throw new SolutionHistoryIdentityError("Invalid submission history owner");
  }
  if (!Array.isArray(value.runs) || value.runs.length > solutionHistoryPageSize) {
    throw new Error("Invalid submission history page");
  }
  const runs = value.runs.map((row): SolutionRun => {
    if (!isObject(row) || typeof row.id !== "string" || !row.id || row.problem !== problem ||
        typeof row.code !== "string" || typeof row.accepted !== "boolean" ||
        typeof row.created_at !== "string" || !Number.isFinite(Date.parse(row.created_at)) ||
        typeof row.source_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(row.source_sha256)) {
      throw new Error("Invalid submission history entry");
    }
    const report = parseReport(row.report);
    if (report.problem !== problem || (report.source_sha256 && report.source_sha256 !== row.source_sha256)) {
      throw new Error("Submission history report does not match its source");
    }
    return {
      id: row.id, problem, created_at: row.created_at, accepted: row.accepted,
      source_sha256: row.source_sha256, code: row.code, report,
    };
  });
  if (new Set(runs.map((run) => run.id)).size !== runs.length) {
    throw new Error("Duplicate submission history entry");
  }
  return runs;
}

type FetchHistory = (path: string, signal: AbortSignal) => Promise<unknown>;

/** One in-memory reader per account/problem view; nothing is cached in browser storage. */
export function createSolutionHistoryReader(
  userId: string,
  problem: string,
  fetchPage: FetchHistory,
  onChange: (state: SolutionHistoryState) => void,
) {
  let state = emptySolutionHistory();
  let active = true;
  let request: AbortController | undefined;
  function update(next: SolutionHistoryState) {
    if (!active) return;
    state = next;
    onChange(state);
  }
  return {
    getState: () => state,
    async loadMore() {
      if (!active || state.loading || !state.hasMore) return;
      const controller = new AbortController();
      request = controller;
      const offset = state.offset;
      update({ ...state, loading: true, error: false });
      try {
        const query = new URLSearchParams({ problem, limit: String(solutionHistoryPageSize), offset: String(offset) });
        const result = await fetchPage(`/api/practice/runs?${query}`, controller.signal);
        if (!active || request !== controller) return;
        const page = parseSolutionHistory(result, userId, problem);
        const ids = new Set(state.runs.map((run) => run.id));
        const nextOffset = offset + page.length;
        update({
          runs: [...state.runs, ...page.filter((run) => !ids.has(run.id))],
          offset: nextOffset,
          hasMore: page.length === solutionHistoryPageSize && nextOffset <= 100_000,
          loading: false,
          error: false,
        });
      } catch (error) {
        if (active && request === controller) {
          // A mismatched session also invalidates already loaded private records.
          update({ ...(error instanceof SolutionHistoryIdentityError ? emptySolutionHistory() : state), loading: false, error: true });
        }
      } finally {
        if (request === controller) request = undefined;
      }
    },
    dispose() {
      active = false;
      request?.abort();
      request = undefined;
    },
  };
}
