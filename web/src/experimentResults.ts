import type { Report } from "./types";
export type ExperimentView = "auto" | "concept" | "run";

export function experimentTests(problemId: string, report?: Report) {
  return report?.problem === problemId
    ? report.tests.filter((test) => test.layer === "T3" && test.artifacts && Object.keys(test.artifacts).length)
    : [];
}
