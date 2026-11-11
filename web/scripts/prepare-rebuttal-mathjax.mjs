import { cp, mkdir, readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

const require = createRequire(import.meta.url);
const packageFile = require.resolve("mathjax/package.json");
const { version } = JSON.parse(await readFile(packageFile, "utf8"));
if (version !== "3.2.2") throw new Error("OpenReview preview requires MathJax 3.2.2");
const destination = resolve("public/vendor/mathjax-3.2.2");
await mkdir(destination, { recursive: true });
await cp(resolve(dirname(packageFile), "es5"), resolve(destination, "es5"), { recursive: true });
await cp(resolve(dirname(packageFile), "LICENSE"), resolve(destination, "LICENSE"));
console.log("Prepared local MathJax 3.2.2, extensions and fonts for Rebuttal preview");
