// Explicit closure from the pinned Pyodide 0.29.4 lock. loadPackage can resolve
// after reporting a failed dependency, while leaving its parent marked loaded.
export const corePackages = ['numpy', 'atomicwrites', 'attrs', 'six', 'more-itertools', 'pluggy', 'py', 'setuptools', 'pyparsing', 'iniconfig', 'exceptiongroup', 'pytest'];

export async function loadJudgePackages(py, needsYaml) {
  const required = needsYaml ? [...corePackages, 'pyyaml'] : corePackages;
  let detail = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      await py.loadPackage(required, {
        errorCallback(message) { detail = String(message).slice(-2000); },
      });
      const missing = required.filter(name => !Object.hasOwn(py.loadedPackages, name));
      if (missing.length) throw Error(`Missing browser packages: ${missing.join(', ')}`);
      py.runPython(`import numpy, pytest${needsYaml ? ', yaml' : ''}`);
      return;
    } catch (error) {
      detail = `${String(error)}${detail ? `\n${detail}` : ''}`.slice(-3000);
    }
  }
  throw Error(`Browser judge dependencies could not be loaded. ${detail}`);
}
