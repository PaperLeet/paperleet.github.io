export function runtimeDiagnostic(error) {
  const message = String(error);
  // Python's root exception is at the end of its traceback. Keep both ends.
  return message.length <= 4000 ? message : `${message.slice(0, 500)}\n…\n${message.slice(-3490)}`;
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
