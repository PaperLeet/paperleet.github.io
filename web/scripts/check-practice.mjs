import assert from "node:assert/strict";
import { createServer } from "vite";
import { checkBrowserWorkspace } from './check-browser-workspace.mjs';
import { checkBrowserLearning } from './check-browser-learning.mjs';
import { checkBrowserNotifications } from './check-browser-notifications.mjs';
import { checkCodeBackup } from './check-code-backup.mjs';
import { checkBrowserDraftSafety } from './check-browser-draft-safety.mjs';
const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const {
    practiceOwner,
    practiceCacheKey,
    mergeWorkspaceSnapshot,
    withPracticeReport,
  } = await server.ssrLoadModule("/src/PracticeWorkspace.tsx");
  assert.equal(
    practiceOwner(undefined, "alice"),
    "unresolved",
    "Unverified deployment must not select legacy local data",
  );
  assert.equal(
    practiceOwner("hosted", "alice", true),
    "unresolved",
    "Changing account must clear visible private state",
  );
  assert.equal(practiceOwner("hosted"), "anonymous");
  assert.equal(practiceOwner("local", "alice"), "local");
  assert.notEqual(
    practiceCacheKey(
      practiceOwner("hosted", "alice"),
      "https://paperleet.test",
    ),
    practiceCacheKey(practiceOwner("hosted", "bob"), "https://paperleet.test"),
  );
  assert.notEqual(
    practiceCacheKey("anonymous", "https://paperleet.test"),
    practiceCacheKey("user:alice", "https://paperleet.test"),
  );
  assert.notEqual(
    practiceCacheKey("user:alice", "https://a.test"),
    practiceCacheKey("user:alice", "https://b.test"),
  );
  const pending = {
    "rl/gae": {
      code: "local edit",
      updated: "later",
      base: "server-v1",
      revision: 1,
    },
  };
  let merged = mergeWorkspaceSnapshot(
    { "rl/gae": { code: "server edit", updated: "server-v2" } },
    pending,
  );
  assert.equal(
    merged.records["rl/gae"].code,
    "local edit",
    "Conflict must preserve the local edit for review",
  );
  assert.equal(
    merged.conflicts["rl/gae"].code,
    "server edit",
    "Conflict must retain the alternate server draft",
  );
  merged = mergeWorkspaceSnapshot(
    { "rl/gae": { code: "local edit", updated: "server-v2" } },
    pending,
  );
  assert.deepEqual(
    merged.pending,
    {},
    "Already-synced pending copies must not replay over another device",
  );
  merged = mergeWorkspaceSnapshot(
    { "rl/gae": { code: "previous", updated: "server-v1" } },
    pending,
  );
  assert.deepEqual(
    merged.conflicts,
    {},
    "Dirty changes against the same base may sync",
  );
  merged = mergeWorkspaceSnapshot(
    {},
    { "rl/gae": { ...pending["rl/gae"], base: "" } },
  );
  assert.deepEqual(
    merged.conflicts,
    {},
    "A new explicit draft may sync without importing old local workspace",
  );
  const problem = { id: "rl/gae", test_count: 1 };
  const report = {
    problem: "rl/gae",
    tests: [{ id: "case", layer: "T0", passed: true }],
    summary: { T0: [1, 1] },
    selected: "all",
  };
  const failed = {
    ...report,
    summary: { T0: [0, 1] },
    tests: [{ ...report.tests[0], passed: false }],
  };
  const saved = {
    code: "edited while judging",
    updated: "latest-draft",
    acceptedReport: report,
  };
  const result = withPracticeReport(saved, problem, failed);
  assert.equal(
    result.code,
    saved.code,
    "A late result must not overwrite edits made while judging",
  );
  assert.equal(
    result.updated,
    saved.updated,
    "A judging response must not replace the draft concurrency version",
  );
  assert.equal(
    result.acceptedReport,
    report,
    "Later failure must preserve an accepted result",
  );
  const snapshot = mergeWorkspaceSnapshot(
    { "rl/gae": { ...saved, report: failed } },
    { "rl/gae": { ...pending["rl/gae"], acceptedReport: failed } },
  );
  assert.equal(
    snapshot.records["rl/gae"].acceptedReport.tests[0].passed,
    true,
    "Browser pending cache must never override server acceptance",
  );
  console.log(
    "PASS private workspace isolation, cross-device conflicts, server acceptance and late-result draft preservation",
  );
  await checkBrowserWorkspace(server);
  await checkBrowserDraftSafety(server);
  await checkBrowserLearning(server);
  await checkBrowserNotifications(server);
  await checkCodeBackup(server);
} finally {
  await server.close();
}
