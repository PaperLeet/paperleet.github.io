import assert from "node:assert/strict";
import { act, createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom" });
try {
  const model = await server.ssrLoadModule("/src/rebuttalModel.ts");
  const { newProject, demoProject, splitReviews, importReviewFile, assembleResponse, countWords, reviewerOverview, saveResponseSnapshot,
    removeConcern, removeExperiment, restoreRemovedContent,
    validateWorkspace, readWorkspace, writeWorkspace, WORKSPACE_KEY } = model;
  const memory = new Map();
  const storage = { getItem: key => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, value) };
  const initial = readWorkspace(storage);
  assert.equal(initial.error, false);
  assert.equal(initial.workspace.projects.length, 1);
  assert.equal(initial.workspace.projects[0].issues.length, 0, "A new workspace must not pretend demo content is user data");
  const project = demoProject();
  const workspace = { version: 1, activeProjectId: project.id, projects: [project] };
  assert(writeWorkspace(storage, workspace));
  assert.deepEqual(readWorkspace(storage).workspace, workspace);
  const legacy = structuredClone(workspace);
  delete legacy.projects[0].createdAt;
  delete legacy.projects[0].archived;
  delete legacy.projects[0].snapshots;
  assert.deepEqual(validateWorkspace(legacy), legacy, "Existing v1 workspaces retain their exact fields without a destructive migration");
  const deletionProject = structuredClone({
    ...project, id: "deletion-test",
    issues: ["concern-a", "concern-b", "concern-c"].map(id => ({ ...project.issues[0], id, response: `Reply for ${id}` })),
    experiments: [
      ["experiment-a", "concern-a"], ["experiment-b1", "concern-b"], ["experiment-c", "concern-c"],
      ["experiment-b2", "concern-b"], ["experiment-a2", "concern-a"],
    ].map(([id, issueId]) => ({ ...project.experiments[0], id, issueId, status: "done", result: `Evidence for ${id}` })),
  });
  const originalDeletionProject = structuredClone(deletionProject);
  const freeze = value => {
    if (value && typeof value === "object") { Object.values(value).forEach(freeze); Object.freeze(value); }
    return value;
  };
  freeze(deletionProject);
  const concernDeletion = removeConcern(deletionProject, "concern-b");
  assert.deepEqual(concernDeletion.project.issues.map(issue => issue.id), ["concern-a", "concern-c"]);
  assert.deepEqual(concernDeletion.project.experiments.map(experiment => experiment.id), ["experiment-a", "experiment-c", "experiment-a2"],
    "Deleting a concern must remove every linked experiment while preserving other concerns' evidence");
  assert.equal(concernDeletion.removed.projectId, deletionProject.id);
  assert.deepEqual(concernDeletion.removed.issue, { value: deletionProject.issues[1], index: 1 });
  assert.deepEqual(concernDeletion.removed.experiments.map(item => item.index), [1, 3]);
  const restoredConcern = restoreRemovedContent(concernDeletion.project, concernDeletion.removed);
  assert.deepEqual(restoredConcern, originalDeletionProject, "Undo restores the concern, response and all experiment evidence in their original positions");
  assert.equal(restoreRemovedContent(restoredConcern, concernDeletion.removed), restoredConcern, "Repeated undo must not duplicate or reorder content");
  assert.deepEqual(deletionProject, originalDeletionProject, "Removal and restoration must not mutate the source project");
  restoredConcern.issues[1].response = "Edited after undo";
  restoredConcern.experiments[1].result = "Edited evidence after undo";
  assert.equal(concernDeletion.removed.issue.value.response, originalDeletionProject.issues[1].response, "Restored drafts must not mutate the undo snapshot");
  assert.equal(concernDeletion.removed.experiments[0].value.result, originalDeletionProject.experiments[1].result);
  assert.equal(restoreRemovedContent(restoredConcern, concernDeletion.removed), restoredConcern, "Undo never overwrites existing edits with older records");
  const experimentDeletion = removeExperiment(deletionProject, "experiment-b1");
  assert.equal(experimentDeletion.project.issues, deletionProject.issues, "Deleting one experiment keeps its concern and response intact");
  assert.deepEqual(experimentDeletion.project.experiments.map(experiment => experiment.id), ["experiment-a", "experiment-c", "experiment-b2", "experiment-a2"]);
  assert.equal(experimentDeletion.removed.issue, undefined);
  assert.deepEqual(restoreRemovedContent(experimentDeletion.project, experimentDeletion.removed), originalDeletionProject);
  for (const remove of [removeConcern, removeExperiment]) {
    const missing = remove(deletionProject, "missing");
    assert.equal(missing.project, deletionProject);
    assert.equal(missing.removed, null, "Deleting a missing record creates no undo entry");
  }
  const otherProject = { ...concernDeletion.project, id: "other-project" };
  assert.equal(restoreRemovedContent(otherProject, concernDeletion.removed), otherProject, "Undo cannot restore content into another paper");
  assert.equal(restoreRemovedContent(concernDeletion.project, experimentDeletion.removed), concernDeletion.project,
    "An experiment cannot be restored after its concern has disappeared");
  const missingReviewer = { ...concernDeletion.project, reviews: [], issues: [], experiments: [] };
  assert.equal(restoreRemovedContent(missingReviewer, concernDeletion.removed), missingReviewer,
    "Undo cannot restore a concern whose reviewer no longer exists or leave orphan experiments");
  const fullIssues = { ...concernDeletion.project, issues: [
    ...concernDeletion.project.issues,
    ...Array.from({ length: 1998 }, (_, index) => ({ ...deletionProject.issues[0], id: `extra-concern-${index}` })),
  ] };
  assert.equal(restoreRemovedContent(fullIssues, concernDeletion.removed), fullIssues, "Undo is atomic when restoring a concern would exceed the 2000-item limit");
  const fillExperiments = (value, count) => ({ ...value, experiments: [
    ...value.experiments,
    ...Array.from({ length: count - value.experiments.length }, (_, index) => ({ ...deletionProject.experiments[0], id: `extra-experiment-${index}` })),
  ] });
  const fullExperiments = fillExperiments(experimentDeletion.project, 2000);
  assert.equal(restoreRemovedContent(fullExperiments, experimentDeletion.removed), fullExperiments, "Experiment undo respects the 2000-item limit");
  const fullConcernExperiments = fillExperiments(concernDeletion.project, 2000);
  assert.equal(restoreRemovedContent(fullConcernExperiments, concernDeletion.removed), fullConcernExperiments,
    "An experiment limit failure must not partially restore its concern");
  const lastSlot = restoreRemovedContent(fillExperiments(experimentDeletion.project, 1999), experimentDeletion.removed);
  assert.equal(lastSlot.experiments.length, 2000, "Restoring exactly to the limit remains supported");
  assert.equal(lastSlot.experiments[1].id, "experiment-b1");
  const legacyDeletion = removeConcern(legacy.projects[0], legacy.projects[0].issues[0].id);
  const legacyRestored = restoreRemovedContent(legacyDeletion.project, legacyDeletion.removed);
  assert.deepEqual(legacyRestored, legacy.projects[0], "Delete and undo preserve older projects without adding new schema fields");
  for (const changed of [concernDeletion.project, experimentDeletion.project, legacyDeletion.project, legacyRestored, lastSlot]) {
    const changedWorkspace = { version: 1, activeProjectId: changed.id, projects: [changed] };
    assert.deepEqual(validateWorkspace(changedWorkspace), changedWorkspace, "Delete and undo remain compatible with validated v1 storage");
  }
  const archived = structuredClone(workspace);
  archived.projects[0].archived = true;
  archived.projects[0].reviews[0].summary = "Author's reviewer summary";
  archived.projects[0].reviews[0].rating = "4: Borderline";
  archived.projects[0].reviews[0].aiSummary = { zh: "审稿人要求补充消融实验。", en: "The reviewer requests an ablation study." };
  assert.deepEqual(validateWorkspace(archived), archived, "Archive and reviewer metadata survive export and reload");
  const before = memory.get(WORKSPACE_KEY);
  assert(!writeWorkspace({ ...storage, setItem() { throw Error("quota"); } }, workspace));
  assert.equal(memory.get(WORKSPACE_KEY), before);
  memory.set(WORKSPACE_KEY, "{broken");
  assert(readWorkspace(storage).error);
  assert.equal(memory.get(WORKSPACE_KEY), "{broken", "Reading a broken save must preserve its original bytes");
  assert(readWorkspace({ getItem() { throw Error("denied"); }, setItem() {} }).error);

  for (const mutate of [
    value => { value.version = 2; },
    value => { value.activeProjectId = "missing"; },
    value => { value.projects.push(value.projects[0]); },
    value => { value.projects[0].issues[0].reviewerId = "missing"; },
    value => { value.projects[0].experiments[0].issueId = "missing"; },
    value => { value.projects[0].issues[0].status = "complete"; },
    value => { value.projects[0].reviews.push(value.projects[0].reviews[0]); },
    value => { value.projects[0].wordLimit = -1; },
    value => { value.projects[0].wordLimit = 1.5; },
    value => { value.projects[0].updatedAt = "yesterday"; },
    value => { value.projects[0].reviews[0].text = "x".repeat(200001); },
    value => { value.projects[0].createdAt = "not a date"; },
    value => { value.projects[0].archived = "yes"; },
    value => { value.projects[0].reviews[0].summary = []; },
    value => { value.projects[0].reviews[0].aiSummary = { en: "Missing Chinese" }; },
    value => { value.projects[0].reviews[0].aiSummary = { zh: " ", en: "Empty Chinese" }; },
    value => { value.projects[0].reviews[0].aiSummary = { zh: "总结", en: "x".repeat(6001) }; },
    value => { value.projects[0].reviews[0].aiSummary = { zh: "总结", en: "Summary", extra: "unrequested" }; },
    value => { value.projects[0].snapshots = [{ id: "v1", createdAt: "bad", label: "test", markdown: "text" }]; },
    value => { value.projects[0].snapshots = Array(101).fill({ id: "v1", createdAt: project.updatedAt, label: "test", markdown: "text" }); },
  ]) {
    const invalid = structuredClone(workspace);
    mutate(invalid);
    assert.throws(() => validateWorkspace(invalid));
  }
  assert.throws(() => validateWorkspace(null));
  assert.throws(() => validateWorkspace({ version: 1, activeProjectId: "x", projects: [] }));

  const reviews = [
    { id: "r1", name: "R1", text: "Summary:\r\nClear paper summary.\r\nStrengths:\r\nThe experiments are impressive.\r\nWeaknesses:\r\n1. Missing ablation.\r\n   Please remove the selection rule.\r\n2. The definition is unclear.\r\n\r\nQuestions:\r\n- Please fix the notation.\r\nRating: 6\r\nConfidence: 4" },
    { id: "r2", name: "R2", text: "优点：\n文章清晰。\n不足：\n1. 缺少基线对比。\n2. 需要多随机种子的方差。\n\n问题：\n如何选取参数？\n评分：5" },
  ];
  const parsed = splitReviews(reviews, "zh");
  assert.equal(parsed.issues.length, 6);
  assert.equal(parsed.experiments.length, 3);
  for (const issue of parsed.issues) {
    const source = reviews.find(review => review.id === issue.reviewerId);
    assert(source.text.includes(issue.quote), "Quote must be an exact substring, including original newlines");
    assert.equal(issue.response, "");
  }
  assert(!parsed.issues.some(issue => /impressive|清晰|Confidence/.test(issue.quote)), "Skip summary, strengths and scores");
  assert(parsed.issues.find(issue => issue.quote.includes("notation")).category === "writing");
  assert(parsed.experiments.every(experiment => experiment.status === "planned" && experiment.result === ""));
  const freeform = splitReviews([{ id: "r3", name: "R3", text: "The definition needs clarification.\n\nPlease evaluate peak memory." }], "en");
  assert.equal(freeform.issues.length, 2);
  assert.equal(freeform.experiments[0].title, "Efficiency and resource cost");
  const multipleQuestions = splitReviews([{ id: "r4", name: "R4", text: "为什么这样选？能否给出定义？\n\n1. A concern\r\n   continued here\n   and here" }], "zh");
  assert.equal(multipleQuestions.issues.length, 3);
  assert.equal(multipleQuestions.issues[2].quote, "1. A concern\r\n   continued here\n   and here");
  assert.equal(splitReviews([{ id: "r5", name: "R5", text: "Weaknesses:\nNone.\nQuestions:\nN/A" }], "en").issues.length, 0,
    "Empty-review placeholders must not become action items");
  const inlineSections = splitReviews([{ id: "r6", name: "R6", text: "Summary: A useful paper.\nStrengths: Good results.\n**Weaknesses:** Missing the ablation.\nQuestions: Why use this objective?" }], "en");
  assert.deepEqual(inlineSections.issues.map(issue => issue.quote), ["Missing the ablation.", "Why use this objective?"],
    "Inline section headings must not cause substantive feedback to disappear");
  const structuredReview = { id: "structured", name: "Reviewer S", text: "Paper Summary:\nThis paper investigates sparse adaptation.\nInformativeness: 2: Useful\nCritical Strengths:\nThe method is simple.\nCritical Weaknesses:\nW1. The matched baseline is missing.\nW2. The definition needs clarification.\nQuestions For Authors:\nWhy is this mask used?\nFlag For Ethics Review: No\nRating: 4: Borderline\nConfidence: 3: Familiar" };
  const structuredIssues = splitReviews([structuredReview], "en").issues;
  assert.deepEqual(structuredIssues.map(issue => issue.title), ["The matched baseline is missing.", "The definition needs clarification.", "Why is this mask used?"]);
  assert.deepEqual(reviewerOverview(structuredReview, structuredIssues), {
    summary: "This paper investigates sparse adaptation.", strengths: "The method is simple.", rating: "4: Borderline", confidence: "3: Familiar",
  }, "Overview fields must remain separate from actionable concerns");
  assert.deepEqual(reviewerOverview({ id: "cn", name: "R", text: "论文总结：方法有意义。\n优点：实现简洁。\n不足：缺少对照。\n评分：5\n置信度：4" }, []),
    { summary: "方法有意义。", strengths: "实现简洁。", rating: "5", confidence: "4" });
  assert.equal(reviewerOverview({ ...structuredReview, summary: "Author-edited summary" }, structuredIssues).summary, "Author-edited summary");
  const noSummary = { id: "r3", name: "R3", text: "Unstructured concern." };
  assert.equal(reviewerOverview(noSummary, [...freeform.issues, ...parsed.issues]).summary, freeform.issues.map(issue => issue.title).join("; "));
  assert.equal(reviewerOverview(noSummary, []).rating, "", "Missing scores must not be inferred");

  const empty = newProject();
  assert.equal(assembleResponse(empty), "");
  const response = assembleResponse(project);
  assert(response.includes("DEMO"));
  assert(!response.includes(project.experiments[0].protocol), "Unrun experiment plans must not become submission evidence");
  assert(!response.includes(project.issues[0].quote), "Unanswered concerns must not become empty submission stubs");
  assert(response.includes(project.issues.find(issue => issue.response).quote));
  assert(!assembleResponse(project, project.reviews[1].id).includes("Algorithm 1"), "Reviewer selection scopes draft export");
  assert.equal(countWords("A clear response."), 3);
  assert.equal(countWords("回答 two concerns"), 4);
  assert.equal(countWords(""), 0);
  assert.throws(() => saveResponseSnapshot(empty), "Empty projects cannot create meaningless history versions");
  const versioned = saveResponseSnapshot(project, "First round");
  assert.equal(project.snapshots.length, 0, "Saving a version must not mutate the original project");
  assert.equal(versioned.snapshots.length, 1);
  assert.equal(versioned.snapshots[0].label, "First round");
  assert.equal(versioned.snapshots[0].markdown, response);
  const revised = structuredClone(versioned);
  revised.issues.find(issue => issue.response).response = "A later reply";
  assert.equal(revised.snapshots[0].markdown, response, "Later changes cannot rewrite a saved response version");
  assert.deepEqual(validateWorkspace({ ...workspace, projects: [versioned] }).projects[0], versioned);
  const secondVersion = saveResponseSnapshot(revised);
  assert.equal(secondVersion.snapshots.length, 2);
  assert.notEqual(secondVersion.snapshots[0].id, secondVersion.snapshots[1].id);
  assert(secondVersion.snapshots[1].markdown.includes("A later reply"));
  assert.throws(() => saveResponseSnapshot({ ...project, snapshots: Array(100).fill(versioned.snapshots[0]) }), "The history cap must not silently delete earlier versions");
  assert.throws(() => validateWorkspace({ ...workspace, projects: [{ ...versioned, snapshots: [versioned.snapshots[0], versioned.snapshots[0]] }] }));

  const legacySnapshot = { id: "legacy-unscoped", createdAt: project.updatedAt, label: "Previous export", markdown: "Legacy source stays unchanged" };
  const scopedHistoryProject = { ...newProject(), id: "scoped-history", reviews: [
    { id: "r1", name: "Reviewer One", text: "First review" }, { id: "r2", name: "Reviewer Two", text: "Second review" },
  ], issues: [
    { ...project.issues[0], id: "history-issue-r1", reviewerId: "r1", response: "Only reviewer one response" },
    { ...project.issues[0], id: "history-issue-r2", reviewerId: "r2", response: "Only reviewer two response" },
  ], snapshots: [legacySnapshot] };
  const reviewerOneV1 = saveResponseSnapshot(scopedHistoryProject, undefined, undefined, "reviewer-r1");
  const reviewerTwoV1 = saveResponseSnapshot(reviewerOneV1, undefined, "Personal edited reviewer two source", "reviewer-r2");
  const renamedHistory = { ...reviewerTwoV1, reviews: reviewerTwoV1.reviews.map(review => review.id === "r1" ? { ...review, name: "Renamed reviewer one" } : review) };
  const reviewerOneV2 = saveResponseSnapshot(renamedHistory, undefined, "Revised reviewer one source", "reviewer-r1");
  assert.deepEqual(reviewerOneV2.snapshots.slice(1).map(version => [version.scope, version.label]), [
    ["reviewer-r1", "Version 1"], ["reviewer-r2", "Version 1"], ["reviewer-r1", "Version 2"],
  ], "Each reviewer starts and advances an independent version sequence");
  assert.equal(reviewerOneV2.snapshots[1].reviewerName, "Reviewer One", "Renaming a reviewer does not rename historical versions");
  assert.equal(reviewerOneV2.snapshots[3].reviewerName, "Renamed reviewer one");
  assert(reviewerOneV2.snapshots[1].markdown.includes("Only reviewer one response"));
  assert(!reviewerOneV2.snapshots[1].markdown.includes("Only reviewer two response"), "An omitted Markdown argument assembles the selected reviewer only");
  assert.equal(reviewerOneV2.snapshots[2].markdown, "Personal edited reviewer two source", "Saving keeps the author's exact selected source");
  assert.deepEqual(scopedHistoryProject.snapshots, [legacySnapshot]);
  const wholePaperV1 = saveResponseSnapshot(reviewerOneV2);
  assert.equal(wholePaperV1.snapshots.at(-1).label, "Version 1", "Legacy unscoped exports are not silently numbered as All-reviewer versions");
  assert.equal(wholePaperV1.snapshots.at(-1).scope, "all");
  assert(!Object.hasOwn(wholePaperV1.snapshots.at(-1), "reviewerName"));
  const historicalWorkspace = { version: 1, activeProjectId: wholePaperV1.id, projects: [wholePaperV1] };
  assert.deepEqual(validateWorkspace(JSON.parse(JSON.stringify(historicalWorkspace))), historicalWorkspace);
  assert.deepEqual(wholePaperV1.snapshots[0], legacySnapshot, "Legacy snapshots round-trip without invented scope or reviewer names");
  const deletedReviewer = { ...wholePaperV1, reviews: wholePaperV1.reviews.filter(review => review.id !== "r1"), issues: wholePaperV1.issues.filter(issue => issue.reviewerId !== "r1") };
  assert.deepEqual(validateWorkspace({ ...historicalWorkspace, projects: [deletedReviewer] }).projects[0].snapshots, wholePaperV1.snapshots,
    "Deleting a reviewer preserves that reviewer's immutable scoped history");
  assert.throws(() => saveResponseSnapshot(deletedReviewer, undefined, "New source", "reviewer-r1"), "A missing reviewer cannot receive a new version");
  assert.throws(() => saveResponseSnapshot(scopedHistoryProject, undefined, "New source", "invalid-scope"));
  const snapshotWorkspace = metadata => ({ ...workspace, projects: [{ ...project, snapshots: [{ ...legacySnapshot, ...metadata }] }] });
  for (const metadata of [
    { scope: null }, { scope: "" }, { scope: "reviewer-" }, { scope: "reviewer-a/b" }, { scope: "reviewer-a b" },
    { scope: "reviewer-审稿人" }, { scope: `reviewer-${"x".repeat(151)}` }, { scope: "all", reviewerName: "Reviewer" },
    { reviewerName: "Reviewer" }, { scope: "reviewer-r1", reviewerName: null },
    { scope: "reviewer-r1", reviewerName: "x".repeat(301) }, { scope: "reviewer-r1", reviewerName: [] },
  ]) assert.throws(() => validateWorkspace(snapshotWorkspace(metadata)), "Invalid history metadata must be rejected");
  const longestHistoryMetadata = { scope: `reviewer-${"x".repeat(150)}`, reviewerName: "名".repeat(300) };
  assert.deepEqual(validateWorkspace(snapshotWorkspace(longestHistoryMetadata)).projects[0].snapshots[0], { ...legacySnapshot, ...longestHistoryMetadata });
  assert.deepEqual(validateWorkspace(snapshotWorkspace({ scope: "reviewer-deleted" })).projects[0].snapshots[0], { ...legacySnapshot, scope: "reviewer-deleted" });

  const imported = importReviewFile({ notes: [
    { id: "review123", content: { reviewer: { value: "Reviewer ABC" }, summary: { value: "A good paper." }, strengths: "Useful experiments.", weaknesses: { value: "1. The ablation is missing." }, questions: { value: "Why use this objective?" }, rating: { value: "6" } } },
    { id: "decision", content: { decision: { value: "Accept" } } },
  ] });
  assert.equal(imported.length, 1);
  assert.equal(imported[0].name, "Reviewer ABC");
  assert.equal(imported[0].id, "review123");
  assert.equal(imported[0].summary, "A good paper.");
  assert.equal(imported[0].strengths, "Useful experiments.");
  assert.equal(imported[0].rating, "6");
  assert.equal(splitReviews(imported, "en").issues.length, 2);
  const importedCritical = importReviewFile({ notes: [{ content: { paper_summary: { value: "A clear summary." }, critical_strengths: { value: "Simple implementation." }, critical_weaknesses: { value: "W1. Missing baseline.\nW2. Notation is unclear." }, rating: { value: 3 }, confidence: { value: "4: Confident" } } }] });
  assert.equal(importedCritical[0].rating, "3");
  assert.equal(importedCritical[0].confidence, "4: Confident");
  assert.equal(splitReviews(importedCritical, "en").issues.length, 2);
  const simple = importReviewFile([{ name: "R1", text: "A question" }, { name: "R2", text: "Another question" }]);
  assert.equal(simple.length, 2);
  assert.notEqual(simple[0].id, simple[1].id);
  const bilingual = { name: "Reviewer B", text: "Weaknesses: Missing ablation.", summary: "A paper summary.",
    aiSummary: { zh: "审稿人要求补充消融实验。", en: "The reviewer requests an ablation study." } };
  const importedBilingual = importReviewFile([bilingual]);
  assert.deepEqual(importedBilingual[0].aiSummary, bilingual.aiSummary, "Bilingual reviewer summaries survive import without replacing source summaries");
  assert.equal(importedBilingual[0].summary, bilingual.summary);
  assert.throws(() => importReviewFile([{ ...bilingual, aiSummary: { zh: "总结" } }]), "Partial bilingual imports must not silently lose a language");
  assert.throws(() => importReviewFile(workspace), "Workspace backups must use the workspace import path");
  assert.throws(() => importReviewFile({ notes: [{ content: { summary: "No concerns" } }] }));
  assert.throws(() => importReviewFile([{ text: "" }]));

  const { default: RebuttalLibrary } = await server.ssrLoadModule("/src/RebuttalLibrary.tsx");
  const renderLibrary = libraryWorkspace => renderToStaticMarkup(createElement(RebuttalLibrary, {
    workspace: libraryWorkspace,
    onOpen() {}, onCreate() {}, onArchive() {}, onImport() {}, onExport() {}, onSample() {},
  }));
  const emptyLibrary = renderLibrary(initial.workspace);
  assert(emptyLibrary.includes("No papers yet"), "A first visit shows an empty library rather than an internal placeholder paper");
  assert(!emptyLibrary.includes("Open Untitled rebuttal"));
  const legacyLibrary = renderLibrary(legacy);
  assert(legacyLibrary.includes(`Open ${project.title}`), "Legacy papers without archive fields remain visible in the active library");
  assert(legacyLibrary.includes("responses ready"));
  const mixedLibrary = renderLibrary({ ...legacy, projects: [...legacy.projects, { ...newProject(), title: "Archived draft", archived: true }] });
  assert(mixedLibrary.includes("Archived"), "The paper library exposes an archive/history entry point");
  assert(!mixedLibrary.includes("Open Archived draft"), "Archived papers do not appear among active work");
  const historyOnlyPaper = { ...newProject(), snapshots: versioned.snapshots };
  const historyLibrary = renderLibrary({ version: 1, activeProjectId: historyOnlyPaper.id, projects: [historyOnlyPaper] });
  assert(historyLibrary.includes("Open Untitled rebuttal"), "A paper with saved history must never be hidden as an unused template");

  // Exercise the hosted editor itself: AI extraction must produce the stored schema, and
  // a temporary membership lock must not re-submit an old workspace on unlock.
  const dom = new JSDOM('<!doctype html><div id="studio-test"></div>', { url: "https://paperleet.test/#/rebuttal/paper_a", pretendToBeVisual: true });
  let delayedAnalysis, delayedSummary;
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    location: dom.window.location, history: dom.window.history, localStorage: dom.window.localStorage,
    HTMLElement: dom.window.HTMLElement, HTMLInputElement: dom.window.HTMLInputElement,
    HTMLTextAreaElement: dom.window.HTMLTextAreaElement, IS_REACT_ACT_ENVIRONMENT: true,
    fetch: async path => {
      if (String(path).endsWith('/api/rebuttal/status')) return new Response(JSON.stringify({ configured: true, model: 'test-model' }));
      if (String(path).endsWith('/api/rebuttal/summarize')) { assert(delayedSummary); return delayedSummary; }
      assert(String(path).endsWith('/api/rebuttal/analyze'), 'The fixture must not send a live request');
      if (delayedAnalysis) return delayedAnalysis;
      return new Response(JSON.stringify({ issues: [
        { reviewerId: 'review_a', title: 'Add the ablation', quote: 'Please add an ablation.', category: 'experiment', priority: 'high', suggestion: 'Compare the components.',
          experiment: { title: 'Component ablation', protocol: 'Remove one component at a time.', metric: 'Accuracy' } },
        { reviewerId: 'review_a', title: 'Clarify notation', quote: 'Please define alpha.', category: 'clarification', priority: 'medium', suggestion: 'Define alpha.' },
      ] }));
    },
  };
  const previousGlobals = new Map(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  let studioRoot;
  try {
    const { createRoot } = await import('react-dom/client');
    const { default: RebuttalStudio } = await server.ssrLoadModule('/src/RebuttalStudio.tsx');
    const source = { ...newProject(), id: 'paper_a', title: 'Hosted paper', reviews: [{ id: 'review_a', name: 'Reviewer A', text: 'Please add an ablation.\nPlease define alpha.' }] };
    const submissions = [];
    const session = { project: source, readOnly: false, saveStatus: 'saved', role: 'owner', onChange: value => submissions.push(value), onBack() {}, onShare() {}, onRefresh() {} };
    studioRoot = createRoot(document.querySelector('#studio-test'));
    const renderStudio = async (overrides = {}, key = 'paper_a:0') => act(async () => studioRoot.render(createElement(RebuttalStudio, { key, hosted: { ...session, ...overrides } })));
    const button = name => [...document.querySelectorAll('button')].find(node => (node.getAttribute('aria-label') || node.textContent.trim()) === name);
    const click = async name => act(async () => { const target = button(name); assert(target, `Missing studio action: ${name}`); target.click(); });
    await renderStudio(); assert.equal(submissions.length, 0, 'Mounting a hosted paper never saves it');
    await click('Reviews'); await click('Extract concerns');
    assert.equal(submissions.length, 1);
    const extracted = submissions[0];
    assert.equal(extracted.issues.length, 2); assert.equal(extracted.experiments.length, 1);
    const issueFields = ['id', 'reviewerId', 'title', 'quote', 'category', 'priority', 'status', 'suggestion', 'outline', 'response'].sort();
    for (const issue of extracted.issues) assert.deepEqual(Object.keys(issue).sort(), issueFields, 'AI-only experiment metadata must not leak into the strict cloud issue schema');
    const plan = extracted.experiments[0];
    assert.equal(plan.issueId, extracted.issues[0].id); assert.equal(plan.protocol, 'Remove one component at a time.');
    assert.equal(plan.status, 'planned'); assert.equal(plan.result, '', 'A proposed experiment has no fabricated results');
    assert.deepEqual(validateWorkspace({ version: 1, activeProjectId: extracted.id, projects: [extracted] }).projects[0], extracted);
    const serverCopy = { ...extracted, updatedAt: '2030-01-01T00:00:00.000Z' };
    await renderStudio({ project: serverCopy, readOnly: true });
    await renderStudio({ project: serverCopy, readOnly: false });
    assert.equal(submissions.length, 1, 'Membership locking and unlocking cannot re-submit the previous workspace');
    await click('Paper settings');
    const titleInput = document.querySelector('.rst-settings input');
    await act(async () => {
      Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(titleInput, 'A new author edit');
      titleInput.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
    });
    assert.equal(submissions.length, 2, 'A real edit after unlock is still delivered');
    assert.equal(submissions[1].title, 'A new author edit');
    await renderStudio({ project: { ...serverCopy, title: 'Explicit server reload' } }, 'paper_a:1');
    assert(document.querySelector('.rst-paper-heading h1').textContent.includes('Explicit server reload'));
    assert.equal(submissions.length, 2, 'A reload mounts the server project without sending an old draft back');
    assert.equal(localStorage.length, 0, 'Hosted edits do not write the local workspace');

    // Revoking edit permission while async work is pending must also discard its late result.
    let resolveAnalysis;
    delayedAnalysis = new Promise(resolve => { resolveAnalysis = resolve; });
    await renderStudio({}, 'paper_a:async-analysis');
    await click('Reviews'); await click('Extract concerns');
    await renderStudio({ readOnly: true }, 'paper_a:async-analysis');
    await act(async () => resolveAnalysis(new Response(JSON.stringify({ issues: [{
      reviewerId: 'review_a', title: 'Late concern', quote: 'Please add an ablation.', category: 'clarification', priority: 'high', suggestion: 'Late suggestion',
    }] }))));
    assert.equal(document.querySelectorAll('.rst-breakdown-item').length, 0);
    assert(button('Extract concerns').disabled && button('Local breakdown').disabled, 'Viewer analysis controls are visibly disabled');
    await renderStudio({ readOnly: false }, 'paper_a:async-analysis');
    assert.equal(submissions.length, 2, 'Unlock cannot later submit an AI result received during readonly mode');
    delayedAnalysis = undefined;

    let resolveSummary;
    delayedSummary = new Promise(resolve => { resolveSummary = resolve; });
    await renderStudio({}, 'paper_a:async-summary'); await click('AI summaries');
    await renderStudio({ readOnly: true }, 'paper_a:async-summary');
    await act(async () => resolveSummary(new Response(JSON.stringify({ summaries: [{ reviewerId: 'review_a', zh: '延迟总结', en: 'Late summary after permission loss' }] }))));
    await renderStudio({ readOnly: false }, 'paper_a:async-summary');
    assert(!document.body.textContent.includes('Late summary after permission loss'));
    assert.equal(submissions.length, 2, 'Late summaries cannot mutate a readonly paper or surface on later unlock');

    await click('Experiments'); await click('Add experiment');
    await renderStudio({ readOnly: true }, 'paper_a:async-summary');
    assert([...document.querySelectorAll('.rst-experiment-create input,.rst-experiment-create select')].every(field => field.disabled), 'A pending experiment form also becomes readonly');
  } finally {
    if (studioRoot) await act(async () => studioRoot.unmount());
    dom.window.close();
    for (const [key, descriptor] of previousGlobals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  }
  console.log("PASS rebuttal parsing, reviewer overviews, source fidelity, experiment plans, reversible deletion, response history, library visibility, validated storage, hosted AI schema and save/reload bridge");
} finally {
  await server.close();
}

await import("./check-rebuttal-assembly-drafts.mjs");
