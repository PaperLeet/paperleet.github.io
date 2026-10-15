// Runs only inside the disposable, opaque-origin browser judge sandbox.
import { calibrationDiagnostic, runtimeDiagnostic, unsupportedReport } from '../public/browser-judge/diagnostics.mjs';
import { syntaxCheck } from '../public/browser-judge/syntax.mjs';
import { loadJudgePackages } from '../public/browser-judge/packages.mjs';
import { calibrationProof, calibrationSummary } from './browserCalibration.ts';
const send = globalThis.postMessage.bind(globalThis);
let started = false;
self.onmessage = async ({ data }) => {
  if (started) return;
  started = true;
  const { base, bundle, source, calibrationKey = '', cachedCalibration } = data;
  const startedAt = performance.now();
  const progress = (stage, detail = {}) => send({ type: 'progress', stage, ...detail });
  const unsupported = (reason, detail = '') => send({ type: 'unsupported', reason, detail });
  let runtimeStage = 'python';
  try {
    progress('loading', { step: 'python' });
    const { loadPyodide } = await import(`${base}browser-runtime/pyodide-0.29.4/pyodide.mjs`);
    const py = await loadPyodide({ indexURL: `${base}browser-runtime/pyodide-0.29.4/`, stdout() {}, stderr() {} });
    runtimeStage = 'syntax';
    progress('loading', { step: 'syntax' });
    py.globals.set('learner_source', source);
    const diagnostic = JSON.parse(py.runPython(syntaxCheck));
    if (diagnostic) return send({ type: 'syntax_error', diagnostic });
    runtimeStage = 'packages';
    progress('loading', { step: 'packages' });
    await loadJudgePackages(py, bundle.version === 3 ? bundle.packages
      : Object.keys(bundle.files || {}).some(name => name.endsWith('/problem.yaml')));
    runtimeStage = 'preparing';
    progress('loading', { step: 'preparing' });
    py.globals.set('bundle_json', JSON.stringify(bundle));
    py.runPython(`import json, inspect\nbundle = json.loads(bundle_json)\nexec(compile(bundle['harness'], 'browser_harness.py', 'exec'))\nprepare(bundle, bundle['adapter'], '/judge')`);
    py.globals.set('_browser_progress', (step, completed, total) => progress(step === 'tests' ? 'running' : 'calibrating', { step, completed, total }));
    py.globals.set('_browser_activity', (step, completed, total, index, test, layer, why, mutant) => progress(step === 'tests' ? 'running' : 'calibrating', {
      step, completed, total, activity: { index, test, layer, why, ...(mutant ? { mutant } : {}) },
    }));
    // Older immutable bundles have a harness without optional callbacks.
    const withProgress = py.runPython("'progress' in inspect.signature(calibrate).parameters and 'progress' in inspect.signature(run_suite).parameters");
    const withActivity = py.runPython("'activity' in inspect.signature(calibrate).parameters and 'activity' in inspect.signature(run_suite).parameters");
    const extra = (withProgress ? ', progress=_browser_progress' : '') + (withActivity ? ', activity=_browser_activity' : '');
    const environment = JSON.stringify([py.version, py.runPython("json.dumps([__import__('sys').version, __import__('numpy').__version__, __import__('sys').byteorder])")]);
    const cached = calibrationProof(cachedCalibration, calibrationKey, bundle, environment);
    const readyAt = performance.now();
    runtimeStage = 'calibration';
    progress('calibrating');
    const calibration = cached ? {compatible:true, ...cached.summary}
      : JSON.parse(py.runPython(`json.dumps(calibrate(bundle, '/judge'${extra}), allow_nan=False)`));
    if (!calibration.compatible) return unsupported('calibration_failed', calibrationDiagnostic(calibration));
    const summary = calibrationSummary(calibration, bundle);
    if (!summary) return unsupported('calibration_failed', 'Incomplete reference/mutation calibration');
    // Sent before ANY learner execution. Later user code cannot replace this proof.
    send({type:'calibrated', proof:{key:calibrationKey,environment,summary}, reused:!!cached});
    const calibratedAt = performance.now();
    runtimeStage = 'tests';
    progress('running');
    const report = JSON.parse(py.runPython(`json.dumps(run_suite(bundle, learner_source, '/judge'${extra}), allow_nan=False)`));
    if (report.error) return unsupported('collection_failed');
    if (unsupportedReport(report)) return unsupported('python_api_unsupported');
    const finishedAt = performance.now();
    send({ type: 'result', report, calibration:summary, measurements:{
      loading_ms:readyAt-startedAt,calibration_ms:calibratedAt-readyAt,
      learner_ms:finishedAt-calibratedAt,total_ms:finishedAt-startedAt,reused:!!cached,
    }});
  } catch (error) {
    unsupported('runtime_failed', runtimeDiagnostic(error, runtimeStage));
  }
};
