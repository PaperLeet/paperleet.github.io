import type { TestResult } from "./types";
export type TestOutcome = "all" | "failed" | "passed";
export function testKey(test: TestResult, index: number) { return `${test.nodeid || test.id}-${index}`; }
export function visibleReportTests(tests: TestResult[], outcome: TestOutcome, layer: string) {
  return tests.map((test, index) => ({ test, key: testKey(test, index) })).filter(({ test }) =>
    (outcome === "all" || test.passed === (outcome === "passed")) && (layer === "all" || test.layer === layer));
}
export function formatTestDuration(seconds: number) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return safe > 0 && safe < .01 ? "<0.01s" : `${safe.toFixed(2)}s`;
}
