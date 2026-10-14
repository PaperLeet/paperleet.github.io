import type { Problem } from "./types";

export async function loadProblemDetail(summary: Problem, baseUrl: string, fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<Problem> {
  if (!summary.detail_url) return summary;
  // Exported content-addressed paths only; do not follow arbitrary catalog URLs.
  if (!/^modules\/\d+-[a-f0-9]{20}\.json$/.test(summary.detail_url)) throw Error("Invalid module detail path");
  const response = await fetcher(`${baseUrl}data/${summary.detail_url}`, { cache: "force-cache", signal });
  if (!response.ok) throw Error("Module details unavailable");
  const detail = await response.json();
  if (detail.id !== summary.id || detail.number !== summary.number || detail.starter !== summary.starter ||
      typeof detail.readme !== "string" || typeof detail.notes !== "string" || !detail.module) {
    throw Error("Module details do not match the catalog");
  }
  return { ...detail, detail_url: summary.detail_url };
}
