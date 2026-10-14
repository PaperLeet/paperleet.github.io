// Runs only inside the disposable, opaque-origin browser judge sandbox.
import { calibrationDiagnostic, runtimeDiagnostic } from './diagnostics.mjs';
import { syntaxCheck } from './syntax.mjs';
const send = globalThis.postMessage.bind(globalThis);
let started = false;
self.onmessage = async ({ data }) => {
  if (started) return;
  started = true;
  const { base, bundle, source } = data;
  const progress = (stage, detail = {}) => send({ type: 'progress', stage, ...detail });
  const unsupported = (reason, detail = '') => send({ type: 'unsupported', reason, detail });
  try {
    progress('loading', { step: 'python' });
    const { loadPyodide } = await import(`${base}browser-runtime/pyodide-0.29.4/pyodide.mjs`);
    const py = await loadPyodide({ indexURL: `${base}browser-runtime/pyodide-0.29.4/`, stdout() {}, stderr() {} });
    progress('loading', { step: 'syntax' });
    py.globals.set('learner_source', source);
    const diagnostic = JSON.parse(py.runPython(syntaxCheck));
    if (diagnostic) return send({ type: 'syntax_error', diagnostic });
    progress('loading', { step: 'packages' });
    const packages = ['numpy', 'pytest'];
    if (Object.keys(bundle.files || {}).some(name => name.endsWith('/problem.yaml'))) packages.push('pyyaml');
    await py.loadPackage(packages);
    progress('loading', { step: 'preparing' });
    py.globals.set('bundle_json', JSON.stringify(bundle));
    py.runPython(`import json, inspect\nbundle = json.loads(bundle_json)\nexec(compile(bundle['harness'], 'browser_harness.py', 'exec'))\nprepare(bundle, bundle['adapter'], '/judge')`);
    py.globals.set('_browser_progress', (step, completed, total) => progress(step === 'tests' ? 'running' : 'calibrating', { step, completed, total }));
    // Older immutable bundles have a harness without optional callbacks.
    const withProgress = py.runPython("'progress' in inspect.signature(calibrate).parameters and 'progress' in inspect.signature(run_suite).parameters");
    const extra = withProgress ? ', progress=_browser_progress' : '';
    progress('calibrating');
    const calibration = JSON.parse(py.runPython(`json.dumps(calibrate(bundle, '/judge'${extra}), allow_nan=False)`));
    if (!calibration.compatible) return unsupported('calibration_failed', calibrationDiagnostic(calibration));
    progress('running');
    const report = JSON.parse(py.runPython(`json.dumps(run_suite(bundle, learner_source, '/judge'${extra}), allow_nan=False)`));
    if (report.error) return unsupported('collection_failed');
    if (report.tests.some((test) => /BrowserUnsupported|ModuleNotFoundError|ImportError/.test(test.message))) return unsupported('python_api_unsupported');
    send({ type: 'result', report, calibration });
  } catch (error) {
    unsupported('runtime_failed', runtimeDiagnostic(error));
  }
};
