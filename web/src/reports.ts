import type { Report, Saved, TestResult } from "./types";
import { parseComparison } from "./comparisons";

const validLayers = new Set(["T0", "T1", "T2", "T3"]);
const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function invalid(field: string): never {
  throw new Error(`不是有效的 PaperLeet report.json：${field}`);
}

function optionalString(value: unknown, field: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") return invalid(field);
  return value;
}

/** Validate both uploaded reports and responses from the local judge before storage. */
export function parseReport(value: unknown): Report {
  if (!isObject(value)) return invalid("报告必须是对象");
  if (typeof value.problem !== "string" || !value.problem.trim())
    return invalid("problem 必须是非空字符串");
  if (!Array.isArray(value.tests)) return invalid("tests 必须是数组");
  const tests: TestResult[] = value.tests.map((row, index) => {
    const field = `tests[${index}]`;
    if (!isObject(row)) return invalid(field);
    if (typeof row.id !== "string" || !row.id.trim())
      return invalid(`${field}.id`);
    if (typeof row.layer !== "string" || !validLayers.has(row.layer))
      return invalid(`${field}.layer`);
    if (typeof row.passed !== "boolean") return invalid(`${field}.passed`);
    if (
      row.duration_s !== undefined &&
      (typeof row.duration_s !== "number" ||
        !Number.isFinite(row.duration_s) ||
        row.duration_s < 0)
    )
      return invalid(`${field}.duration_s`);
    if (row.artifacts !== undefined && !isObject(row.artifacts))
      return invalid(`${field}.artifacts`);
    const nodeid = optionalString(row.nodeid, `${field}.nodeid`);
    if (nodeid !== undefined && !nodeid.trim())
      return invalid(`${field}.nodeid`);
    return {
      id: row.id,
      layer: row.layer,
      passed: row.passed,
      comparison: row.passed ? undefined : parseComparison(row.comparison),
      nodeid,
      why: optionalString(row.why, `${field}.why`),
      message: optionalString(row.message, `${field}.message`),
      duration_s: row.duration_s === undefined ? 0 : row.duration_s,
      artifacts: row.artifacts as Record<string, unknown> | undefined,
    };
  });
  // Parameterized tests can share the same original name; nodeid identifies a case.
  const nodeids = tests.flatMap((test) =>
    test.nodeid === undefined ? [] : [test.nodeid],
  );
  if (new Set(nodeids).size !== nodeids.length)
    return invalid("重复的 test nodeid");
  if (!isObject(value.summary)) return invalid("summary 必须是对象");
  const summary: Report["summary"] = {};
  for (const [layer, pair] of Object.entries(value.summary)) {
    if (
      !validLayers.has(layer) ||
      !Array.isArray(pair) ||
      pair.length !== 2 ||
      !pair.every(
        (n) => typeof n === "number" && Number.isSafeInteger(n) && n >= 0,
      ) ||
      pair[0] > pair[1]
    )
      return invalid(`summary.${layer}`);
    summary[layer] = [pair[0], pair[1]];
  }
  for (const layer of validLayers) {
    const rows = tests.filter((test) => test.layer === layer);
    const expected = [rows.filter((test) => test.passed).length, rows.length];
    const actual = summary[layer] ?? [0, 0];
    if (actual[0] !== expected[0] || actual[1] !== expected[1])
      return invalid(`summary.${layer} 与实际测试行不一致`);
  }
  let mutation: Report["mutation"];
  if (value.mutation === null) mutation = null;
  else if (value.mutation !== undefined) {
    if (!isObject(value.mutation)) return invalid("mutation 必须是对象或 null");
    mutation = Object.fromEntries(
      Object.entries(value.mutation).map(([id, caught]) => {
        if (
          !Array.isArray(caught) ||
          !caught.every((test) => typeof test === "string")
        )
          return invalid(`mutation.${id}`);
        return [id, [...caught]];
      }),
    );
  }
  const sourceHash = optionalString(value.source_sha256, "source_sha256");
  const runId = optionalString(value.run_id, "run_id");
  if (runId !== undefined && !/^[a-f0-9]{32}$/.test(runId))
    return invalid("run_id 必须是提交记录的 ID");
  if (sourceHash !== undefined && !/^[a-f0-9]{64}$/.test(sourceHash))
    return invalid("source_sha256 必须是 SHA-256 十六进制字符串");
  return {
    problem: value.problem,
    ...(runId ? { run_id: runId } : {}),
    source_sha256: sourceHash,
    runtime: optionalString(value.runtime, "runtime"),
    tests,
    summary,
    mutation,
    impl: optionalString(value.impl, "impl"),
    created_at: optionalString(value.created_at, "created_at"),
    error: optionalString(value.error, "error"),
  };
}

/** Recover valid drafts while discarding malformed history and mismatched reports. */
export function parseSaved(value: unknown): Record<string, Saved> {
  if (!isObject(value)) return {};
  const entries: [string, Saved][] = [];
  for (const [id, entry] of Object.entries(value)) {
    if (!isObject(entry) || !id.trim()) continue;
    const saved: Saved = {};
    if (typeof entry.code === "string") saved.code = entry.code;
    if (typeof entry.updated === "string") saved.updated = entry.updated;
    try {
      const report = parseReport(entry.report);
      if (report.problem === id) saved.report = report;
    } catch {
      /* One broken report must not prevent recovery of other drafts. */
    }
    try {
      const accepted = parseReport(entry.acceptedReport);
      if (accepted.problem === id && !accepted.error && !accepted.mutation &&
          accepted.tests.length > 0 && accepted.tests.every((test) => test.passed))
        saved.acceptedReport = accepted;
    } catch {
      /* Completion is validated against the current module by status(). */
    }
    // Retain successes from the original storage format before the latest
    // report is replaced or the editor is reset. status() checks full coverage.
    if (!saved.acceptedReport && saved.report && !saved.report.error &&
        !saved.report.mutation && saved.report.tests.length > 0 &&
        saved.report.tests.every((test) => test.passed))
      saved.acceptedReport = saved.report;
    if (saved.code !== undefined || saved.report || saved.acceptedReport) entries.push([id, saved]);
  }
  return Object.fromEntries(entries);
}

export function decodeProblemHash(hash: string): string {
  try {
    // Decode only the path: an escaped '&' in a search must stay query data.
    const raw = hash.replace(/^#\/?/, "");
    const query = raw.indexOf('?');
    return query < 0 ? decodeURIComponent(raw) : decodeURIComponent(raw.slice(0, query)) + raw.slice(query);
  } catch {
    return "";
  }
}
