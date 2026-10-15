import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';

const manifest = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
const initial = new Set();
function visit(key) {
  if (initial.has(key)) return;
  assert(manifest[key], `Missing static import: ${key}`);
  initial.add(key);
  for (const dependency of manifest[key].imports || []) visit(dependency);
}
visit('index.html');
for (const key of ['src/PythonEditor.tsx', 'src/RichText.tsx', 'src/Community.tsx', 'src/Account.tsx', 'src/Submissions.tsx', 'src/Lab.tsx', 'src/highlight.ts', 'src/browserLearningController.ts', 'src/SolutionHistory.tsx', 'src/AccountLogout.tsx', 'src/WorkspaceBackup.tsx', 'src/libraryPosition.ts', 'src/QueueAdmissionNotice.tsx', 'src/browserJudge.ts', 'src/ReportView.tsx', 'src/ModuleNavigator.tsx']) {
  assert(manifest[key]?.isDynamicEntry, `${key} must remain a deferred entry`);
  assert(!initial.has(key), `${key} must not load with the paper library`);
}
for (const key of ['src/queuedJudgeController.ts','src/queueAdmission.ts']) {
  if (process.env.VITE_PUBLIC_SITE === '1') assert(!manifest[key], `${key} must be removed from the browser-only edition`);
  else {
    assert(manifest[key]?.isDynamicEntry, `${key} must load only when server judging is needed`);
    assert(!initial.has(key), `${key} must not load with the paper library`);
  }
}
for (const key of ['src/ResetCodeButton.tsx', 'src/Settings.tsx', 'src/Achievements.tsx', 'src/AchievementObserver.tsx', 'src/BrowserResultActions.tsx', 'src/JudgeRuntimeDetails.tsx', 'src/ReportWorkflow.tsx', 'src/WorkspaceLayoutMenu.tsx', 'src/PracticeGuideContent.tsx', 'src/HeaderAccountMenu.tsx']) {
  assert(manifest[key]?.isDynamicEntry && !initial.has(key), `${key} must load with practice rather than the library`);
}
const files = [...initial].map(key => manifest[key].file);
assert(files.every(file => !/\/(editor|markdown|Community|Lab|PythonEditor|RichText|SolutionHistory)-/.test(file)), 'Optional runtime dependencies must not be pulled back into the shell');
const raw = files.reduce((sum, file) => sum + fs.statSync(`dist/${file}`).size, 0);
const gzip = files.reduce((sum, file) => sum + zlib.gzipSync(fs.readFileSync(`dist/${file}`), { level: 9 }).length, 0);
assert(raw <= 400 * 1024 && gzip <= 130 * 1024, `Initial JavaScript exceeds the 400 KiB raw / 130 KiB gzip budget: ${raw} / ${gzip}`);
const css = new Set([...initial].flatMap(key => manifest[key].css || []));
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('KaTeX_Main-Regular')), 'Math font styles must load with rich text rather than the homepage');
console.log(`PASS initial build graph: ${files.length} JS assets, ${raw} bytes raw / ${gzip} bytes gzip; editor, rich text, community, accounts, submissions, labs, highlighting and solution history deferred`);
