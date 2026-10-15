import type { Problem } from "./types";

const failedDetails = new Set<string>();
const MAX_FAILED_DETAILS = 128;
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");
const optionalText = (value: unknown) => value === undefined || typeof value === "string";

// Compare against the index without depending on JSON object key order. Recursion
// is bounded by the small, known index fields, not by unknown extra response data.
function sameValue(expected: unknown, actual: unknown): boolean {
  if (expected === actual) return true;
  if (Array.isArray(expected)) return Array.isArray(actual) && expected.length === actual.length &&
    expected.every((item, index) => sameValue(item, actual[index]));
  if (!record(expected) || !record(actual)) return false;
  const keys = Object.keys(expected);
  return keys.length === Object.keys(actual).length && keys.every((key) =>
    Object.prototype.hasOwnProperty.call(actual, key) && sameValue(expected[key], actual[key]));
}

function validDetail(value: unknown, summary: Problem): value is Problem {
  if (!record(value) || typeof value.readme !== "string" || typeof value.notes !== "string" ||
      !optionalText(value.solution)) return false;
  const fields = ["id", "number", "paper_id", "short_id", "title", "difficulty", "type", "parent",
    "module", "estimated_minutes", "tests", "starter", "browser_judge"] as const;
  if (!fields.every((field) => sameValue(summary[field], value[field]))) return false;
  const paper = value.paper;
  if (!record(paper) || typeof paper.title !== "string" || !paper.title.trim() ||
      !["url", "venue", "arxiv"].every((key) => optionalText(paper[key])) ||
      !(optionalText(paper.authors) || strings(paper.authors)) ||
      !(paper.anchors === undefined || strings(paper.anchors))) return false;
  if (!Array.isArray(value.functions) || !value.functions.every((item) => record(item) &&
      typeof item.name === "string" && typeof item.signature === "string")) return false;
  if (value.translations !== undefined) {
    if (!record(value.translations)) return false;
    const english = value.translations.en;
    if (english !== undefined && (!record(english) || typeof english.title !== "string" ||
        !optionalText(english.readme) || !optionalText(english.notes))) return false;
    const expectedTitle = summary.translations?.en?.title;
    if (expectedTitle !== undefined && (!record(english) || english.title !== expectedTitle)) return false;
  } else if (summary.translations?.en) return false;
  return value.vizData === undefined || value.vizData === null || record(value.vizData);
}

function checkAbort(signal?: AbortSignal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException("Module loading aborted", "AbortError");
}

export async function loadProblemDetail(summary: Problem, baseUrl: string, fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<Problem> {
  checkAbort(signal);
  if (!summary.detail_url) return summary;
  // Exported content-addressed paths only; do not follow arbitrary catalog URLs.
  if (!/^modules\/\d+-[a-f0-9]{20}\.json$/.test(summary.detail_url)) throw Error("Invalid module detail path");
  const url = `${baseUrl}data/${summary.detail_url}`;
  try {
    // Successful immutable details keep their normal cache path. A failed body
    // must not trap the Retry button on a cached HTTP 200 error indefinitely.
    const response = await fetcher(url, { cache: failedDetails.has(url) ? "reload" : "force-cache", signal });
    if (!response.ok) throw Error("Module details unavailable");
    const detail: unknown = await response.json();
    checkAbort(signal);
    if (!validDetail(detail, summary)) throw Error("Module details do not match the catalog");
    failedDetails.delete(url);
    return { ...detail, detail_url: summary.detail_url };
  } catch (error) {
    if (!signal?.aborted) {
      failedDetails.delete(url);
      failedDetails.add(url);
      if (failedDetails.size > MAX_FAILED_DETAILS) failedDetails.delete(failedDetails.values().next().value!);
    }
    throw error;
  }
}
