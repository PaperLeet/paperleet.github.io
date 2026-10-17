import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';
import {loadEnv} from 'vite';

const manifest = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
const initial = new Set();
function visit(key) {
  if (initial.has(key)) return;
  assert(manifest[key], `Missing static import: ${key}`);
  initial.add(key);
  for (const dependency of manifest[key].imports || []) visit(dependency);
}
visit('index.html');
for (const key of ['src/ReadingTabControls.tsx', 'src/PaperDetail.tsx', 'src/PythonEditor.tsx', 'src/RichText.tsx', 'src/Community.tsx', 'src/Account.tsx', 'src/Submissions.tsx', 'src/Lab.tsx', 'src/highlight.ts', 'src/browserLearningController.ts', 'src/browserWorkspaceController.ts', 'src/SolutionHistory.tsx', 'src/AccountLogout.tsx', 'src/WorkspaceBackup.tsx', 'src/libraryPosition.ts', 'src/QueueAdmissionNotice.tsx', 'src/browserJudge.ts', 'src/ReportView.tsx', 'src/WorkspaceTestResults.tsx', 'src/ModuleNavigator.tsx', 'src/InterviewWritingGuide.tsx', 'src/CommunityWritingGuide.tsx', 'src/catalogData.ts', 'src/hostedDraftCache.ts', 'src/SubmissionResultPage.tsx', 'src/OnlinePresence.tsx', 'src/ResizableSplit.tsx', 'src/Messages.tsx']) {
  assert(manifest[key]?.isDynamicEntry, `${key} must remain a deferred entry`);
  assert(!initial.has(key), `${key} must not load with the paper library`);
}
const bellEntry = Object.entries(manifest).find(([, entry]) => entry.name === 'InboxBell' && entry.isDynamicEntry);
const environment = loadEnv('production', process.cwd());
const onlineAccounts = environment.VITE_APP_EDITION === 'hosted' || (environment.VITE_PUBLIC_SITE === '1' && environment.VITE_HOSTED_APP_AVAILABLE === '1');
if (onlineAccounts) {
  assert(manifest['src/InboxPanel.tsx']?.isDynamicEntry && !initial.has('src/InboxPanel.tsx'), 'The notification panel must remain deferred');
  assert(bellEntry && !initial.has(bellEntry[0]), 'The notification bell must remain deferred, including when Rollup shares its icon chunk');
} else {
  assert(!bellEntry && !manifest['src/InboxPanel.tsx'], 'Unavailable notification controls must be removed from offline editions');
}
for (const key of ['src/queuedJudgeController.ts','src/queueAdmission.ts']) {
  if (process.env.VITE_PUBLIC_SITE === '1') assert(!manifest[key], `${key} must be removed from the browser-only edition`);
  else {
    assert(manifest[key]?.isDynamicEntry, `${key} must load only when server judging is needed`);
    assert(!initial.has(key), `${key} must not load with the paper library`);
  }
}
for (const key of ['src/ResetCodeButton.tsx', 'src/Settings.tsx', 'src/Achievements.tsx', 'src/AchievementObserver.tsx', 'src/BrowserResultActions.tsx', 'src/JudgeRuntimeDetails.tsx', 'src/WorkspaceTestResults.tsx', 'src/WorkspaceLayoutMenu.tsx', 'src/HeaderAccountMenu.tsx', 'src/Interviews.tsx', 'src/ProblemCompanyTags.tsx']) {
  assert(manifest[key]?.isDynamicEntry && !initial.has(key), `${key} must load with practice rather than the library`);
}
const files = [...initial].map(key => manifest[key].file);
assert(files.every(file => !/\/(editor|markdown|Community|Lab|PythonEditor|RichText|SolutionHistory)-/.test(file)), 'Optional runtime dependencies must not be pulled back into the shell');
const raw = files.reduce((sum, file) => sum + fs.statSync(`dist/${file}`).size, 0);
const gzip = files.reduce((sum, file) => sum + zlib.gzipSync(fs.readFileSync(`dist/${file}`), { level: 9 }).length, 0);
assert(raw <= 400 * 1024 && gzip <= 130 * 1024, `Initial JavaScript exceeds the 400 KiB raw / 130 KiB gzip budget: ${raw} / ${gzip}`);
const css = new Set([...initial].flatMap(key => manifest[key].css || []));
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('community-code-manual')), 'Enhanced community code controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('discussion-copy')), 'Discussion sharing controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('failure-frame-link')), 'Interactive traceback controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('has-live-preview')), 'Live writing controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('data-writing-title')), 'Composer focus controls must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.writing-disclosure-trigger')), 'Composer focus styling must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('data-writing-title')), 'Deferred composers must include explicit entry focus controls');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Numbered steps')), 'Structural writing tools must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Numbered steps')), 'Deferred writing must include structural formatting controls');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('discussion-reply-search')), 'Reply search controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('post-card-unsaved-notice')), 'Feed bookmark controls must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.post-card-unsaved-notice')), 'Feed bookmark feedback styles must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('post-card-unsaved-notice')), 'Deferred community cards must include inline save recovery');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('preparation-concept-picker')), 'Study concept search controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('preparation-concept-navigation')), 'Sequential study navigation must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.preparation-concept-navigation')), 'Sequential study navigation styles must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.preparation-concept-picker')), 'Study concept search styles must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.discussion-reading.is-compact')), 'Compact feed reading styles must remain deferred');
assert(Object.values(manifest).some(entry => (entry.css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.discussion-reading.is-compact'))), 'Deferred readers must include the shared compact outline layout');
for (const marker of ['Where does DAPS apply measurement conditioning', 'Which two parameters does MoGe alignment fit']) assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes(marker)), 'Individual exercise handbook content must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.discussion-reply-search')), 'Reply search styles must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.has-live-preview')), 'Live writing layout must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.failure-frame-link')), 'Interactive traceback styles must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.discussion-link-fallback')), 'Discussion sharing styles must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.community-code-manual')), 'Community code reading styles must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('community-code-manual')), 'Deferred community Markdown must include the manual code-copy recovery');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('KaTeX_Main-Regular')), 'Math font styles must load with rich text rather than the homepage');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.python-definition-nav')), 'Python navigation styles must stay deferred with the editor');
assert((manifest['src/PythonEditor.tsx'].css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.python-definition-nav')), 'The enhanced editor must include its navigation layout');
console.log(`PASS initial build graph: ${files.length} JS assets, ${raw} bytes raw / ${gzip} bytes gzip; editor, rich text, community, accounts, submissions, labs, highlighting and solution history deferred`);
