export type BrowserStage = 'loading' | 'calibrating' | 'running';
export type BrowserStep = 'bundle' | 'python' | 'syntax' | 'packages' | 'preparing' | 'reference' | 'mutants' | 'tests';
export type BrowserActivity = { index: number; test: string; layer: 'T0' | 'T1' | 'T2' | 'T3'; why: string; mutant?: string };
export type BrowserProgress = { stage: BrowserStage; step?: BrowserStep; completed?: number; total?: number; activity?: BrowserActivity };
const stages: BrowserStage[] = ['loading', 'calibrating', 'running'];
const steps: Record<BrowserStage, BrowserStep[]> = { loading: ['bundle', 'python', 'syntax', 'packages', 'preparing'], calibrating: ['reference', 'mutants'], running: ['tests'] };

function parseActivity(value: unknown, step: BrowserStep, completed: number, total: number, tests: number): BrowserActivity | undefined {
  if (!value || typeof value !== 'object' || completed >= total) return;
  const row = value as Record<string, unknown>;
  if (!Number.isSafeInteger(row.index) || Number(row.index) < 0 || Number(row.index) >= tests
    || step !== 'mutants' && row.index !== completed) return;
  const text = (input: unknown, limit: number): input is string => typeof input === 'string' && input.trim().length > 0 && input.length <= limit && !/[\u0000-\u001f\u007f]/.test(input);
  if (!text(row.test, 200) || !text(row.why, 320) || typeof row.layer !== 'string' || !['T0','T1','T2','T3'].includes(row.layer)) return;
  if (step === 'mutants' ? !text(row.mutant, 120) : row.mutant !== undefined) return;
  return { index: Number(row.index), test: row.test, layer: row.layer as BrowserActivity['layer'], why: row.why,
    ...(step === 'mutants' ? { mutant: row.mutant as string } : {}) };
}

/** Ignore malformed/stale advisory messages. They never determine acceptance or reset deadlines. */
export function advanceBrowserProgress(previous: BrowserProgress, message: unknown, totals: { tests: number; mutants: number }): BrowserProgress | undefined {
  if (!message || typeof message !== 'object') return;
  const value = message as Record<string, unknown>;
  const stage = value.stage as BrowserStage;
  const phase = stages.indexOf(stage), before = stages.indexOf(previous.stage);
  if (phase < before || phase < 0) return;
  // Older worker scripts publish only phase transitions.
  if (value.step === undefined) return phase > before ? { stage } : undefined;
  const step = value.step as BrowserStep;
  const index = steps[stage].indexOf(step);
  if (index < 0) return;
  const next: BrowserProgress = { stage, step };
  if (stage !== 'loading') {
    const total = step === 'mutants' ? totals.mutants : totals.tests;
    if (!Number.isSafeInteger(value.completed) || !Number.isSafeInteger(value.total) || value.total !== total
      || total < 1 || Number(value.completed) < 0 || Number(value.completed) > total) return;
    next.completed = Number(value.completed); next.total = total;
    const activity = parseActivity(value.activity, step, next.completed, total, totals.tests);
    if (activity) next.activity = activity;
  }
  if (phase === before && previous.step) {
    const oldIndex = steps[stage].indexOf(previous.step);
    if (index < oldIndex || index === oldIndex && (stage === 'loading' || next.completed! < (previous.completed ?? -1))) return;
    if (index === oldIndex && next.completed === previous.completed
      && (!next.activity || next.activity.index <= (previous.activity?.index ?? -1)
        || previous.activity && next.activity.mutant !== previous.activity.mutant)) return;
  }
  return next;
}

export function browserProgressText(value: BrowserProgress, en = false) {
  const count = value.completed !== undefined && value.total !== undefined ? ` ${value.completed} / ${value.total}` : '';
  const labels: Record<BrowserStep, [string, string]> = {
    bundle: ['加载题目测试包…', 'Loading problem test bundles…'],
    python: ['准备浏览器 Python…', 'Preparing browser Python…'],
    syntax: ['检查 Python 语法…', 'Checking Python syntax…'],
    packages: ['加载 NumPy 与测试工具…', 'Loading NumPy and test tools…'],
    preparing: ['准备隔离的测试环境…', 'Preparing the isolated test environment…'],
    reference: [`校验参考测试${count}`, `Checking reference tests${count}`],
    mutants: [`校验错误实现${count}`, `Checking mutation cases${count}`],
    tests: [`已完成${count} 项测试`, `Tests completed${count}`],
  };
  if (value.step) return labels[value.step][en ? 1 : 0];
  return value.stage === 'calibrating' ? (en ? 'Validating reference and mutation cases…' : '校验参考实现与错误实现…')
    : value.stage === 'running' ? (en ? 'Running the complete suite in your browser…' : '浏览器正在执行完整测试…')
    : (en ? 'Loading browser Python…' : '加载浏览器 Python 运行时…');
}
