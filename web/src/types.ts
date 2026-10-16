import { parseReport } from "./reports";
import type { TestComparison } from "./comparisons";
import type { BrowserBudgets } from "./browserBudgets";
import type { BrowserRunDetails } from "./browserRunDetails";

export type Category = { id: string; name: string; name_en: string };

export type Paper = {
  kind?: "course";
  id: string;
  number: number;
  title: string;
  short_title: string;
  authors: string | string[];
  venue?: string;
  year?: number;
  url: string;
  arxiv?: string;
  tags: string[];
  category: string;
  summary: string;
  summary_en: string;
  scope: string;
  scope_en: string;
  module_ids: string[];
};

export type Problem = {
  detail_url?: string;
  browser_judge?: { bundle_url: string; backend: "numpy-wasm" | "numpy-autograd-wasm"; budgets?: BrowserBudgets };
  id: string;
  number: number;
  paper_id: string;
  module: {
    goal: string;
    goal_en: string;
    scope: string;
    scope_en: string;
    required_functions: string[];
    provided_functions: string[];
    optional: boolean;
  };
  short_id?: string;
  translations?: { en?: { title: string; readme: string; notes: string } };
  title: string;
  difficulty: string;
  type?: string;
  parent?: string | null;
  paper: {
    title: string;
    url?: string;
    authors?: string | string[];
    venue?: string;
    arxiv?: string;
    anchors?: string[];
  };
  functions: { name: string; signature: string }[];
  conventions?: Record<string, unknown>;
  estimated_minutes?: number;
  tests?: Record<string, number>;
  readme: string;
  starter: string;
  notes: string;
  solution?: string;
  vizData?: Record<string, unknown> | null;
};
export type TestResult = {
  comparison?: TestComparison;
  id: string;
  nodeid?: string;
  layer: string;
  passed: boolean;
  why?: string;
  message?: string;
  duration_s?: number;
  artifacts?: Record<string, unknown>;
};
export type Report = {
  browser_run?: BrowserRunDetails;
  run_id?: string;
  source_sha256?: string;
  runtime?: string;
  error?: string;
  problem: string;
  impl?: string;
  created_at?: string;
  summary: Record<string, [number, number]>;
  tests: TestResult[];
  mutation?: Record<string, string[]> | null;
};
// Only browserLearningView sets this projection marker. JSON imports and account
// responses cannot manufacture historical browser completion through a flag.
export const browserLearningPassed = Symbol('browserLearningPassed');
export type Saved = { code?: string; report?: Report; acceptedReport?: Report; updated?: string; [browserLearningPassed]?: true };
export const layers = ["T0", "T1", "T2", "T3"];
export const layerLabels = ["接口契约", "参考对齐", "数学性质", "真实行为"];
export function isPassed(p: Pick<Problem, 'id' | 'tests'>, r?: Report) {
  try {
    const report = parseReport(r);
    if (
      report.error ||
      report.mutation ||
      report.problem !== p.id ||
      !report.tests.length ||
      report.tests.some((test) => !test.passed)
    )
      return false;
    return Object.entries(p.tests || report.summary).every(
      ([layer, expected]) => {
        const required = typeof expected === "number" ? expected : expected[1];
        return (
          layers.includes(layer) &&
          Number.isSafeInteger(required) &&
          required >= 0 &&
          report.tests.filter((test) => test.layer === layer && test.passed)
            .length >= required
        );
      },
    );
  } catch {
    return false;
  }
}
export function status(p: Problem, s: Saved = {}) {
  return s[browserLearningPassed] || isPassed(p, s.acceptedReport) || isPassed(p, s.report)
    ? "done"
    : (s.code !== undefined && s.code !== p.starter) || s.report
      ? "started"
      : "todo";
}
