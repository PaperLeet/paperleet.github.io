// Runs only inside the disposable, opaque-origin browser judge sandbox.
import { calibrationDiagnostic } from './diagnostics.mjs';
const send = globalThis.postMessage.bind(globalThis);
let started = false;
self.onmessage = async ({ data }) => {
  if (started) return;
  started = true;
  const { base, bundle, source } = data;
  const progress = (stage) => send({ type: 'progress', stage });
  const unsupported = (reason, detail = '') => send({ type: 'unsupported', reason, detail });
  try {
    progress('loading');
    const { loadPyodide } = await import(`${base}browser-runtime/pyodide-0.29.4/pyodide.mjs`);
    const py = await loadPyodide({ indexURL: `${base}browser-runtime/pyodide-0.29.4/`, stdout() {}, stderr() {} });
    await py.loadPackage(['numpy', 'pytest']);
    py.globals.set('bundle_json', JSON.stringify(bundle));
    py.globals.set('learner_source', source);
    py.runPython(`import json\nbundle = json.loads(bundle_json)\nexec(compile(bundle['harness'], 'browser_harness.py', 'exec'))\nprepare(bundle, bundle['adapter'], '/judge')`);
    progress('calibrating');
    const calibration = JSON.parse(py.runPython("json.dumps(calibrate(bundle, '/judge'), allow_nan=False)"));
    if (!calibration.compatible) return unsupported('calibration_failed', calibrationDiagnostic(calibration));
    progress('running');
    const report = JSON.parse(py.runPython("json.dumps(run_suite(bundle, learner_source, '/judge'), allow_nan=False)"));
    if (report.error) return unsupported('collection_failed');
    if (report.tests.some((test) => /BrowserUnsupported|ModuleNotFoundError|ImportError/.test(test.message))) return unsupported('python_api_unsupported');
    send({ type: 'result', report, calibration });
  } catch (error) {
    unsupported('runtime_failed', String(error).slice(0, 500));
  }
};
