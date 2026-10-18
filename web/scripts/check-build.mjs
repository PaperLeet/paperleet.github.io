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
for (const key of ['src/ReadingTabControls.tsx', 'src/BrowserRunProgressDetails.tsx', 'src/PaperDetail.tsx', 'src/PythonEditor.tsx', 'src/RichText.tsx', 'src/Community.tsx', 'src/Account.tsx', 'src/Submissions.tsx', 'src/Lab.tsx', 'src/highlight.ts', 'src/browserLearningController.ts', 'src/browserWorkspaceController.ts', 'src/SolutionHistory.tsx', 'src/AccountLogout.tsx', 'src/WorkspaceBackup.tsx', 'src/libraryPosition.ts', 'src/QueueAdmissionNotice.tsx', 'src/browserJudge.ts', 'src/ReportView.tsx', 'src/WorkspaceTestResults.tsx', 'src/ModuleNavigator.tsx', 'src/InterviewWritingGuide.tsx', 'src/CommunityWritingGuide.tsx', 'src/catalogData.ts', 'src/hostedDraftCache.ts', 'src/SubmissionResultPage.tsx', 'src/OnlinePresence.tsx', 'src/ResizableSplit.tsx', 'src/Messages.tsx', 'src/DraftCenter.tsx', 'src/SubmissionCompletionNotice.tsx']) {
  // Rollup gives a shared static/dynamic community chunk a generated manifest key.
  const candidates=manifest[key]?[[key,manifest[key]]]:key==='src/Community.tsx'?Object.entries(manifest).filter(([,entry])=>entry.name==='Community'&&entry.isDynamicEntry):[];
  assert(candidates.length===1&&candidates[0][1].isDynamicEntry, `${key} must remain a unique deferred entry`);
  assert(!initial.has(candidates[0][0]), `${key} must not load with the paper library`);
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
for (const key of ['src/ResetCodeButton.tsx', 'src/Settings.tsx', 'src/Achievements.tsx', 'src/AchievementObserver.tsx', 'src/BrowserResultActions.tsx', 'src/JudgeRuntimeDetails.tsx', 'src/WorkspaceTestResults.tsx', 'src/WorkspaceLayoutMenu.tsx', 'src/HeaderAccountMenu.tsx', 'src/Interviews.tsx', 'src/ProblemCompanyTags.tsx', 'src/ProblemOverview.tsx']) {
  assert(manifest[key]?.isDynamicEntry && !initial.has(key), `${key} must load with practice rather than the library`);
}
const administration=Object.entries(manifest).filter(([,entry])=>entry.name==='AdminWorkspace'&&entry.isDynamicEntry);
assert(administration.length===1&&!initial.has(administration[0][0]), 'Administrator workspace must remain uniquely deferred');
const moderation=Object.entries(manifest).filter(([,entry])=>entry.name==='ModerationQueue'&&entry.isDynamicEntry);
assert(moderation.length===1&&!initial.has(moderation[0][0]), 'Administrator report review must stay deferred');
const history=Object.entries(manifest).filter(([,entry])=>entry.name==='ReportHistory'&&entry.isDynamicEntry);
assert(history.length===1&&!initial.has(history[0][0]), 'Private report history must stay uniquely deferred');
const comparison=Object.entries(manifest).filter(([,entry])=>entry.name==='ModerationSnapshotComparison'&&entry.isDynamicEntry);
assert(comparison.length===1&&!initial.has(comparison[0][0]), 'Optional moderator snapshot comparison must stay uniquely deferred');
const files = [...initial].map(key => manifest[key].file);
const reportPreview=Object.entries(manifest).filter(([,entry])=>entry.name==='InterviewSourcePreview'&&entry.isDynamicEntry);
assert(reportPreview.length===1&&!initial.has(reportPreview[0][0]), 'Source experience preview must remain a unique deferred entry');
assert(manifest['src/MarkdownSectionNav.tsx']?.isDynamicEntry&&!initial.has('src/MarkdownSectionNav.tsx'), 'Optional writing section navigation must remain deferred');
assert(manifest['src/ProblemSolutionPanel.tsx']?.isDynamicEntry&&!initial.has('src/ProblemSolutionPanel.tsx'), 'Per-problem solution navigation must remain deferred');
assert(manifest['src/EditorTools.tsx']?.isDynamicEntry&&!initial.has('src/EditorTools.tsx'), 'Optional editor controls must remain deferred');
assert(manifest['src/LibraryOverview.tsx']?.isDynamicEntry&&!initial.has('src/LibraryOverview.tsx'), 'Practice overview and company interview preparation must remain deferred');
const studyLists=manifest['src/StudyLists.tsx']?[['src/StudyLists.tsx',manifest['src/StudyLists.tsx']]]:Object.entries(manifest).filter(([,entry])=>entry.name==='StudyLists'&&entry.isDynamicEntry);
assert(studyLists.length===1&&studyLists[0][1].isDynamicEntry&&!initial.has(studyLists[0][0]), 'Private study-list reading and editing must remain uniquely deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Could not confirm your private study lists.')), 'Private study-list stores must not grow the library shell');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Company interview practice')), 'Library interview preparation must not grow the initial shell');
assert(files.every(file => !/\/(editor|markdown|Community|Lab|PythonEditor|RichText|SolutionHistory)-/.test(file)), 'Optional runtime dependencies must not be pulled back into the shell');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Your implementation goal')), 'Problem overview and statistics rendering must stay deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Reference reading actions')), 'Reference reading controls must remain deferred with problem solutions');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('This submission is ready to review.')), 'Completion presentation must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('No headings match.')), 'Reader heading filters must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('copying includes the full code.')), 'Long code preview presentation must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Updates available')), 'Inbox reading continuity presentation must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Copy study checklist')), 'Interview batch sharing must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Your report history could not be confirmed.')), 'Private report history data validation must remain deferred');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Read both complete sources')), 'Moderator snapshot comparison must not grow the library shell');
for (const marker of ['Daily limit reached (', 'cannot be changed again until']) assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes(marker)), 'Publication and profile-limit wording must load only with the relevant feature or API rejection');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('createImageBitmap')), 'Avatar image editing must stay deferred with account settings, independently of public avatar URL validation');
const raw = files.reduce((sum, file) => sum + fs.statSync(`dist/${file}`).size, 0);
const gzip = files.reduce((sum, file) => sum + zlib.gzipSync(fs.readFileSync(`dist/${file}`), { level: 9 }).length, 0);
assert(raw <= 400 * 1024 && gzip <= 130 * 1024, `Initial JavaScript exceeds the 400 KiB raw / 130 KiB gzip budget: ${raw} / ${gzip}`);
const css = new Set([...initial].flatMap(key => manifest[key].css || []));
for(const selector of ['.admin-workspace','.community-tag-picker','.moderation-layout','.post-report-dialog','.report-history-card','.moderation-comparison']){
  assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes(selector)), 'Report interfaces must stay out of initial styles');
  assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes(selector))), 'Report interfaces must ship their deferred styles');
}
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Reports and content snapshots')), 'Moderator evidence reading must remain deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.study-list-popover')), 'Private study-list presentation must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.study-list-popover'))), 'Deferred study-list views must include their styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.library-interview-guide')), 'Library interview preparation styles must remain deferred');
assert((manifest['src/LibraryOverview.tsx'].css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.library-interview-guide')), 'Deferred overview must include interview preparation styles');
assert([...css].some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.description-scroll[hidden]')), 'Initial workspace styles must hide inactive retained reading panels');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Discussions for this problem')), 'In-place problem discussions must remain deferred with community reading');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-discussion-header')), 'In-place discussion header styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-discussion-header'))), 'Deferred community views must include in-place discussion styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Your solution was published.')), 'Solution publication feedback must remain deferred with the community');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.solution-publication-notice')), 'Solution publication styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.solution-publication-notice'))), 'Deferred community views must include solution publication feedback styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Loading NumPy and test tools')), 'Detailed browser progress text must stay deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.browser-run-activity')), 'Detailed browser activity styles must stay deferred');
assert((manifest['src/BrowserRunProgressDetails.tsx'].css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.browser-run-activity')), 'Deferred progress must include activity styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Concept list display')), 'Interview list disclosure must stay deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-ranking-display')), 'Interview list display styles must stay deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-ranking-display'))), 'Interview lists must include display controls styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Automatic copy is unavailable. The complete source')), 'Full source copy recovery must stay deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.editor-copy-manual')), 'Source copy recovery styles must stay deferred');
assert((manifest['src/EditorTools.tsx'].css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.editor-copy-manual')), 'Deferred editor controls must include recovery styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-batch-share')), 'Batch sharing styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-batch-share'))), 'Interview batches must include sharing styles');
if (onlineAccounts) {
  assert(manifest['src/InterviewBatchSave.tsx']?.isDynamicEntry&&!initial.has('src/InterviewBatchSave.tsx'), 'Private batch saving must load only after an explicit visit');
  const dependencies=new Set();
  function visitBatch(key){if(dependencies.has(key))return;dependencies.add(key);for(const dependency of manifest[key].imports||[])visitBatch(dependency);}
  visitBatch('src/InterviewBatchSave.tsx');
  assert([...dependencies].every(key=>!['Interviews','InterviewComposer','Community'].includes(manifest[key].name)), 'Private batch saving must not import its parent page or public publishing interfaces');
}
else assert(!manifest['src/InterviewBatchSave.tsx'], 'Unavailable private batch editing must be removed from offline editions');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Save these ')), 'Private batch editing must stay out of the initial library shell');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-batch-save-confirmed')), 'Private batch saving styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-batch-save-confirmed')))===onlineAccounts, 'Batch saving styles must be present only in online editions');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.inbox-list-refresh')), 'Inbox reading continuity styles must remain deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.community-code-disclosure')), 'Code preview styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.community-code-disclosure'))), 'Long code readers must include disclosure styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.discussion-outline-filter')), 'Directory search styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.discussion-outline-filter'))), 'Long article readers must include heading filter styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.submission-completion-notice')), 'Completion styles must remain deferred');
assert((manifest['src/SubmissionCompletionNotice.tsx'].css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.submission-completion-notice')), 'The completion widget must include its styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.reference-reader')), 'Reference reading styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.reference-reading-actions'))), 'Problem solutions must include reference reading styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-interview-panel')), 'Practice experience preview styles must stay deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-interview-panel'))), 'Practice company evidence must include its preview styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.markdown-section-nav')), 'Writing section navigation styles must stay deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.markdown-section-controls'))), 'Writing editors must include section navigation styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Pick up a post, solution or interview.')), 'Writing draft inventory must remain deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.draft-center')), 'Writing draft styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.draft-center-card'))), 'The deferred writing draft center must include its styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Insert current code')), 'Practice code insertion controls must stay deferred with solution writing');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.solution-code-draft')), 'Solution code import styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.solution-code-draft'))), 'Deferred solution writers must include code insertion styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Practice progress for this experience')), 'Experience progress UI must remain deferred with report reading');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-report-progress')), 'Report practice progress styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-report-progress'))), 'Deferred report readers must include linked-practice progress styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Comment changes saved.')), 'Inline comment editing must remain deferred with community reading');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.card-comment-editor')), 'Inline comment editing styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.card-comment-editor'))), 'Deferred community readers must include edit styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Quote comment from')), 'Inline quote actions must remain deferred with community reading');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.reply-quote-notice')), 'Quote feedback styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.reply-quote-notice'))), 'Deferred comment readers must include quote feedback styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Add concepts from this comparison')), 'Comparison practice controls must remain deferred with the interview views');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Company comparison choices')), 'Company comparison must stay deferred with interview preparation');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Linked practice counts could not be confirmed.')), 'Company practice progress must stay deferred with interview preparation');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.company-directory-practice')), 'Company practice progress styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.company-directory-practice'))), 'Deferred interview preparation must include personal company progress styles');

assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.company-comparison')), 'Company comparison styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.company-comparison'))), 'Deferred interview preparation must include company comparison styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('post-search-match')), 'Search highlighting must remain deferred with community reading');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.post-search-match')), 'Search highlight styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.post-search-match'))), 'Deferred community views must include search highlight styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('Post publication period')), 'Publication period controls must remain deferred');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.community-period')), 'Publication period styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.community-period'))), 'Deferred community views must include publication period styles');
assert(files.every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('interview-source-preview-toggle')), 'Source previews must remain deferred with interview preparation');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-source-preview')), 'Source preview styles must remain deferred');
assert(Object.values(manifest).some(entry=>(entry.css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.interview-source-preview'))), 'Deferred interview preparation must include source preview styles');
assert([...css].every(file=>!fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-solution-tabs')), 'Solution navigation styles must remain deferred');
assert((manifest['src/ProblemSolutionPanel.tsx'].css||[]).some(file=>fs.readFileSync(`dist/${file}`,'utf8').includes('.problem-solution-tabs')), 'The solution panel must include its navigation styles');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('community-code-manual')), 'Enhanced community code controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('discussion-copy')), 'Discussion sharing controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('failure-frame-link')), 'Interactive traceback controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Search test details')), 'Test-detail search controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Jump to experience page')), 'Interview page-jump controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Jump to list page')), 'Community page-jump controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Company directory order')), 'Company directory controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Interview outcome')), 'Interview outcome controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('data-interview-facet')), 'Related interview facet links must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('data-interview-facet')), 'Deferred interview metadata must include related experience links');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('View interview experiences')), 'Profile interview controls must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('View interview experiences')), 'Deferred profiles must include interview contributions');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Interview outcome')), 'Deferred interviews must include outcome filtering');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Company directory order')), 'Deferred interviews must include company-directory ordering');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.interview-company-directory')), 'Company-directory styles must remain deferred');
assert(Object.values(manifest).some(entry => (entry.css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.interview-company-directory'))), 'Deferred interviews must include company-directory styles');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Jump to list page')), 'Deferred community views must include page navigation');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.page-jump')), 'Shared page-jump styles must remain deferred');
assert(Object.values(manifest).some(entry => (entry.css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.page-jump'))), 'Deferred list views must include page-jump styles');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.interview-page-jump')), 'Interview page-jump styles must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Jump to experience page')), 'Deferred interviews must include direct page navigation');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.report-test-search')), 'Test-detail search styles must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Search test details')), 'Deferred report views must include test-detail search');
assert(Object.values(manifest).some(entry => (entry.css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.report-test-search'))), 'Deferred report views must include test-detail search styles');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('has-live-preview')), 'Live writing controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('data-writing-title')), 'Composer focus controls must remain deferred');
assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('Copy submitted code manually')), 'Private submitted-source controls must remain deferred');
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.submitted-source')), 'Submitted-source reader styles must remain deferred');
assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes('Copy submitted code manually')), 'Deferred submission views must include the manual source-copy field');
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
for (const marker of ['community-tag-browser','Tag filtering is unavailable']) {
  assert(files.every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes(marker)), 'Tag discovery must remain deferred');
  assert(Object.values(manifest).some(entry => entry.file.endsWith('.js') && fs.readFileSync(`dist/${entry.file}`, 'utf8').includes(marker)), 'Deferred community views must include tag discovery');
}
assert([...css].every(file => !fs.readFileSync(`dist/${file}`, 'utf8').includes('.community-tag-browser')), 'Tag discovery styles must remain deferred');
assert(Object.values(manifest).some(entry => (entry.css || []).some(file => fs.readFileSync(`dist/${file}`, 'utf8').includes('.community-tag-browser'))), 'Deferred community views must include tag discovery styles');
console.log(`PASS initial build graph: ${files.length} JS assets, ${raw} bytes raw / ${gzip} bytes gzip; editor, rich text, community, accounts, submissions, labs, highlighting and solution history deferred`);
