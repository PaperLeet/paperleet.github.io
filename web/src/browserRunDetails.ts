export type BrowserMeasurements = { loading_ms: number; calibration_ms: number; learner_ms: number; total_ms: number; reused: boolean };
/** Historical display data, never an executable calibration proof. */
export type BrowserRunDetails = { version: 1; calibration: { reference_passed: number; mutants_caught: number; reused: boolean }; measurements?: BrowserMeasurements };
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

export function browserMeasurements(value: unknown, maximum = 300_000): BrowserMeasurements | undefined {
  if (!record(value) || typeof value.reused !== 'boolean') return;
  const keys = ['loading_ms','calibration_ms','learner_ms','total_ms'] as const;
  if (!keys.every(key => typeof value[key] === 'number' && Number.isFinite(value[key]) && Number(value[key]) >= 0 && Number(value[key]) <= maximum)) return;
  const result = Object.fromEntries(keys.map(key => [key, value[key]])) as Omit<BrowserMeasurements,'reused'>;
  if (Math.abs(result.loading_ms + result.calibration_ms + result.learner_ms - result.total_ms) > 1) return;
  return { ...result, reused:value.reused };
}

export function browserRunDetails(value: unknown): BrowserRunDetails | undefined {
  if (!record(value) || value.version !== 1 || !record(value.calibration)) return;
  const calibration = value.calibration;
  if (!Number.isSafeInteger(calibration.reference_passed) || Number(calibration.reference_passed) < 1 || Number(calibration.reference_passed) > 100_000
      || !Number.isSafeInteger(calibration.mutants_caught) || Number(calibration.mutants_caught) < 0 || Number(calibration.mutants_caught) > 1000
      || typeof calibration.reused !== 'boolean') return;
  // Completed end-to-end timings include asset loading before the worker's
  // deadline begins. This display bound does not change any judge deadline.
  const measured = browserMeasurements(value.measurements, 3_600_000);
  return { version:1, calibration:{reference_passed:Number(calibration.reference_passed),mutants_caught:Number(calibration.mutants_caught),reused:calibration.reused},
    ...(measured?.reused === calibration.reused ? {measurements:measured} : {}) };
}

