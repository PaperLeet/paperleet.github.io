import type { Report, TestResult } from "./types";
export type ExperimentView = "auto" | "concept" | "run";

/** Keep the parameter identity when several experiments share a test function. */
export function reportTestLabel(test: Pick<TestResult,'id'|'nodeid'>) {
  if (!test.nodeid?.endsWith(']')) return test.id;
  if (test.nodeid.startsWith(`${test.id}[`)) return test.nodeid;
  const start = test.nodeid.indexOf(`::${test.id}[`);
  return start < 0 ? test.id : test.nodeid.slice(start + 2);
}

export function experimentTests(problemId: string, report?: Report) {
  return report?.problem === problemId
    ? report.tests.filter((test) => test.layer === "T3" && test.artifacts && Object.keys(test.artifacts).length)
    : [];
}
