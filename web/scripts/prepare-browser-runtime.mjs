import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pinnedFile } from "./pinned-runtime-file.mjs";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(web, "node_modules/pyodide");
const output = path.join(web, "public/browser-runtime/pyodide-0.29.4");
await fs.mkdir(output, { recursive: true });
for (const name of ["pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) {
  await fs.copyFile(path.join(source, name), path.join(output, name));
}
const lock = JSON.parse(await fs.readFile(path.join(source, "pyodide-lock.json"), "utf8"));
const names = new Set();
function include(name) {
  if (names.has(name)) return;
  const pkg = lock.packages[name];
  if (!pkg) throw Error(`Unknown Pyodide package: ${name}`);
  names.add(name);
  for (const dependency of pkg.depends) include(dependency);
}
include("numpy"); include("pytest"); include("pyyaml");
await Promise.all([...names].map((name) => {
  const pkg = lock.packages[name];
  return pinnedFile(`https://cdn.jsdelivr.net/pyodide/v0.29.4/full/${pkg.file_name}`, path.join(output, pkg.file_name), pkg.sha256);
}));
console.log(`Browser runtime ready: Pyodide 0.29.4, NumPy ${lock.packages.numpy.version}, pytest ${lock.packages.pytest.version}; validated browser modules use CPU/WASM`);
