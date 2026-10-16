/** Public compatibility evidence only. No learner source, reports or Python state. */
export type CalibrationSummary = { reference_passed: number; mutants_caught: Record<string, string[]> };
export type CalibrationProof = { key: string; environment: string; summary: CalibrationSummary };
export { browserMeasurements } from './browserRunDetails';
export type { BrowserMeasurements } from './browserRunDetails';
type Bundle = { tests?: unknown; mutants?: unknown };
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

/** A reference must pass every maintained test; every exact mutant must fail assertions. */
export function calibrationSummary(value: unknown, bundle: Bundle): CalibrationSummary | undefined {
  if (!record(value) || !record(bundle.tests) || !record(bundle.mutants)) return;
  const counts = Object.values(bundle.tests);
  if (!counts.length || !counts.every(count => Number.isSafeInteger(count) && Number(count) >= 0)) return;
  const tests = counts.reduce<number>((sum, count) => sum + Number(count), 0);
  const names = Object.keys(bundle.mutants);
  if (tests < 1 || value.reference_passed !== tests || !record(value.mutants_caught)
      || Object.keys(value.mutants_caught).length !== names.length) return;
  const caught: Record<string, string[]> = {};
  for (const name of names) {
    if (!Object.hasOwn(value.mutants_caught, name)) return;
    const failures = value.mutants_caught[name];
    if (!Array.isArray(failures) || !failures.length || failures.length > tests
        || !failures.every(test => typeof test === 'string' && test.length > 0 && test.length <= 500)) return;
    Object.defineProperty(caught, name, {value:[...failures], enumerable:true, configurable:true, writable:true});
  }
  return { reference_passed: tests, mutants_caught: caught };
}

export function calibrationProof(value: unknown, key: string, bundle: Bundle, environment?: string): CalibrationProof | undefined {
  if (!key || !record(value) || value.key !== key || typeof value.environment !== 'string'
      || !value.environment.length || value.environment.length > 1000 || environment !== undefined && value.environment !== environment) return;
  const summary = calibrationSummary(value.summary, bundle);
  return summary ? { key, environment:value.environment, summary } : undefined;
}

/** This cache dies on refresh. Its lifetime never renews when reused. */
export function createCalibrationCache(now = () => performance.now(), wall = Date.now) {
  const records = new Map<string, {proof:CalibrationProof; created:number; wall:number}>();
  return {
    get(key:string, bundle:Bundle) {
      const entry = records.get(key);
      if (!entry) return;
      const elapsed=now()-entry.created, wallElapsed=wall()-entry.wall;
      if (![elapsed,wallElapsed].every(value=>Number.isFinite(value)&&value>=0&&value<15*60_000)) { records.delete(key); return; }
      const proof = calibrationProof(entry.proof, key, bundle);
      records.delete(key);
      if (proof) records.set(key, entry);
      return proof;
    },
    put(key:string, bundle:Bundle, value:unknown) {
      const proof = calibrationProof(value, key, bundle);
      if (!proof) return false;
      records.delete(key); records.set(key, {proof, created:now(), wall:wall()});
      while (records.size > 8) records.delete(records.keys().next().value!);
      return true;
    },
    remove(key:string) { records.delete(key); },
  };
}
