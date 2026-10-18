export function runtimeDiagnostic(error, stage) {
  const phases = {python: 'Python runtime', syntax: 'Syntax check', packages: 'NumPy and test packages',
    preparing: 'Test environment', calibration: 'Reference and mutation calibration', tests: 'Learner tests'};
  const prefix = Object.hasOwn(phases, stage) ? `Stage: ${phases[stage]}\n` : '';
  const message = prefix + String(error);
  // Python's root exception is at the end of its traceback. Keep both ends.
  return message.length <= 4000 ? message : `${message.slice(0, 500)}\n…\n${message.slice(-3490)}`;
}

// New harnesses report actual exception types. Retain cached immutable harnesses
// without mistaking an assertion that quotes an API name for incompatibility.
export function unsupportedReport(report) {
  return report.tests.some(test => Object.hasOwn(test, 'unsupported_api')
    ? test.unsupported_api === true
    : /BrowserUnsupported|ModuleNotFoundError|ImportError/.test(test.message));
}

// Calibration runs only public reference/mutant code, never the learner source.
export function calibrationDiagnostic(calibration) {
  const lines = [String(calibration?.reason || 'unknown_calibration_failure')];
  if (calibration?.mutant) lines.push(`mutant: ${calibration.mutant}`);
  const report = calibration?.reference || calibration?.report;
  for (const row of Array.isArray(report?.tests) ? report.tests : []) {
    if (!row.passed) lines.push(`${row.id}: ${String(row.message || row.outcome || 'failed').slice(-1000)}`);
  }
  if (report?.error) lines.push(String(report.error).slice(-1000));
  return lines.join('\n').slice(0, 4000);
}
