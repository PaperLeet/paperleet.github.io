export type BrowserBudgets = { calibration: number; run: number; total: number };

export function parseBrowserBudgets(value: unknown): BrowserBudgets | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return;
  const fields = value as Record<string, unknown>;
  const keys = ['calibration', 'run', 'total'];
  if (Object.keys(fields).length !== keys.length || !keys.every(key =>
    Number.isSafeInteger(fields[key]) && Number(fields[key]) > 0 && Number(fields[key]) <= 300_000)) return;
  const budgets = fields as BrowserBudgets;
  if (Math.max(budgets.calibration, budgets.run) > budgets.total) return;
  return {calibration:budgets.calibration, run:budgets.run, total:budgets.total};
}

/** New exports carry validated per-module budgets. Cached v1/v2 exports retain their caps. */
export function browserBudget(problem: {id: string; browser_judge?: {budgets?: BrowserBudgets}}, exported?: unknown): BrowserBudgets {
  return parseBrowserBudgets(exported) || parseBrowserBudgets(problem.browser_judge?.budgets)
    || (problem.id === 'diffusion/flow_matching'
      ? {calibration:240_000,run:60_000,total:300_000}
      : {calibration:90_000,run:30_000,total:120_000});
}
