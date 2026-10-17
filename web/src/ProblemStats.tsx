import { useCallback, useEffect, useRef, useState } from "react";
import { publicSite } from "./deployment";
import { apiFetch } from "./apiClient";
import { subscribePracticeChanges } from "./practiceChanges";

export type ProblemStatistics = {
  attempted_users: number;
  solved_users: number;
  total_submissions: number;
  accepted_submissions: number;
  failed_submissions: number;
  acceptance_rate: number | null;
};
export type ProblemStatsMap = Record<string, ProblemStatistics>;
export const COMMUNITY_STATS_REFRESH_MS = 60 * 60 * 1000;
export const COMMUNITY_STATS_FOCUS_MS = 30_000;

export function totalCommunitySubmissions(stats: ProblemStatsMap | undefined): number | undefined {
  if (!stats) return undefined;
  const total = Object.values(stats).reduce((sum, entry) => sum + entry.total_submissions, 0);
  return Number.isSafeInteger(total) && total >= 0 ? total : undefined;
}

const emptyStats: ProblemStatistics = {
  attempted_users: 0,
  solved_users: 0,
  total_submissions: 0,
  accepted_submissions: 0,
  failed_submissions: 0,
  acceptance_rate: null,
};

// An absent entry means zero submissions only after the public aggregate loaded.
export function getProblemStats(stats: ProblemStatsMap | undefined, id: string) {
  return stats ? stats[id] ?? emptyStats : undefined;
}

function parseStats(value: unknown): ProblemStatsMap {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Invalid practice statistics");
  }
  const countFields = ["attempted_users", "solved_users", "total_submissions", "accepted_submissions", "failed_submissions"] as const;
  for (const entry of Object.values(value)) {
    if (!entry || typeof entry !== "object" || countFields.some((field) => !Number.isSafeInteger(entry[field]) || entry[field] < 0) ||
      !(entry.acceptance_rate === null || (typeof entry.acceptance_rate === "number" && Number.isFinite(entry.acceptance_rate) && entry.acceptance_rate >= 0 && entry.acceptance_rate <= 100))) {
      throw new Error("Invalid practice statistics");
    }
  }
  return value as ProblemStatsMap;
}

export function useProblemStats(refreshKey?: string) {
  const [stats, setStats] = useState<ProblemStatsMap>();
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);
  const lastRefresh = useRef(-Infinity);
  const refresh = useCallback(async () => {
    if (publicSite) { setLoading(false); setAvailable(false); setError(null); return; }
    lastRefresh.current = Date.now();
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    const timeout = window.setTimeout(() => controller.abort(), 12_000);
    try {
      const response = await apiFetch("/api/practice/stats", { signal: controller.signal, credentials: "omit", cache: "no-store" });
      if (!response.ok) throw new Error("Practice statistics are unavailable");
      const data = await response.json();
      if (request.current !== controller) return;
      if (typeof data?.available !== "boolean") throw new Error("Invalid practice statistics");
      const next = data.available ? parseStats(data.stats) : undefined;
      setStats(next);
      setAvailable(data.available);
    } catch {
      if (request.current !== controller) return;
      setStats(undefined);
      setAvailable(null);
      setError("Practice statistics are unavailable");
    } finally {
      window.clearTimeout(timeout);
      if (request.current === controller) {
        setLoading(false);
        request.current = null;
      }
    }
  }, []);
  useEffect(() => {
    void refresh();
    return () => {
      request.current?.abort();
      request.current = null;
    };
  }, [refresh, refreshKey]);
  useEffect(() => {
    if (publicSite) return;
    let nextRefresh = Date.now() + COMMUNITY_STATS_REFRESH_MS;
    let timer: number | undefined;
    let changed = false;
    function sync(revalidate = false) {
      window.clearTimeout(timer);
      if (document.visibilityState !== 'visible' || !navigator.onLine) return;
      if (changed || Date.now() >= nextRefresh || (revalidate && Date.now() - lastRefresh.current >= COMMUNITY_STATS_FOCUS_MS)) {
        changed = false;
        nextRefresh = Date.now() + COMMUNITY_STATS_REFRESH_MS;
        void refresh();
      }
      timer = window.setTimeout(() => sync(), Math.max(0, nextRefresh - Date.now()));
    }
    const resume = () => sync(true);
    const unsubscribe = subscribePracticeChanges(change => {
      if (change.kind !== 'submission') return;
      changed = true;
      sync();
    });
    sync();
    document.addEventListener('visibilitychange', resume);
    window.addEventListener('focus', resume);
    window.addEventListener('online', resume);
    window.addEventListener('offline', resume);
    return () => {
      unsubscribe();
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', resume);
      window.removeEventListener('focus', resume);
      window.removeEventListener('online', resume);
      window.removeEventListener('offline', resume);
    };
  }, [refresh]);
  return { stats, totalSubmissions: totalCommunitySubmissions(stats), refresh, loading, error, available };
}

export {ProblemStats, ProblemStatsSummary} from "./ProblemStatsView";
