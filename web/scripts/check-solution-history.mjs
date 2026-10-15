import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { createSolutionHistoryReader, parseSolutionHistory, SolutionHistoryIdentityError } = await server.ssrLoadModule("/src/solutionHistoryData.ts");
  const problem = "rl/gae";
  const source = "# 原始提交\r\ndef solve():\r\n    return 42\r\n";
  function run(id) {
    return {
      id: String(id), problem, created_at: "2026-10-14T12:00:00+00:00", accepted: true,
      source_sha256: "a".repeat(64), code: source,
      report: { problem, source_sha256: "a".repeat(64), summary: { T0: [1, 1] },
        tests: [{ id: "test_answer", layer: "T0", passed: true }] },
    };
  }
  const page = (runs, user_id = "alice") => ({ user_id, runs });
  const firstPage = Array.from({ length: 30 }, (_, id) => run(id));
  const calls = [];
  const reader = createSolutionHistoryReader("alice", problem, async (path) => {
    const query = new URL(path, "https://paperleet.test").searchParams;
    calls.push(Number(query.get("offset")));
    assert.equal(query.get("problem"), problem, "Each page must request the selected problem");
    assert.equal(query.get("limit"), "30");
    return page(query.get("offset") === "0" ? firstPage : [run(29), run(30)]);
  }, () => {});
  await reader.loadMore();
  assert.equal(reader.getState().runs[0].code, source, "Unicode, line endings and whitespace must remain exactly as submitted");
  assert.equal(reader.getState().runs[0].report.tests[0].passed, true);
  assert.equal(reader.getState().hasMore, true, "A full page must allow access to older submissions");
  await reader.loadMore();
  assert.deepEqual(calls, [0, 30], "Pagination must advance by raw server rows");
  assert.equal(reader.getState().runs.length, 31, "A new submission between pages must not duplicate an existing row");
  assert.equal(reader.getState().offset, 32);
  assert.equal(reader.getState().hasMore, false);
  await reader.loadMore();
  assert.equal(calls.length, 2, "Exhausted history must not issue extra requests");

  assert.throws(() => parseSolutionHistory(page([run(1)], "bob"), "alice", problem), /owner/);
  assert.throws(() => parseSolutionHistory({ runs: [run(1)] }, "alice", problem), /owner/,
    "History without a verified owner must not expose code");
  assert.throws(() => parseSolutionHistory(page([{ ...run(1), problem: "other/problem" }]), "alice", problem), /entry/);
  assert.throws(() => parseSolutionHistory(page([{ ...run(1), source_sha256: "b".repeat(64) }]), "alice", problem), /source/);

  let fail = true;
  let retryCount = 0;
  const retryReader = createSolutionHistoryReader("alice", problem, async () => {
    retryCount++;
    if (fail) throw new Error("offline");
    return page([run(1)]);
  }, () => {});
  await retryReader.loadMore();
  assert.equal(retryReader.getState().error, true);
  assert.equal(retryReader.getState().offset, 0, "Failed requests must not skip history");
  fail = false;
  await retryReader.loadMore();
  assert.equal(retryReader.getState().error, false);
  assert.equal(retryCount, 2);
  assert.equal(retryReader.getState().runs.length, 1);

  let respond;
  let requestSignal;
  let requestCount = 0;
  const published = [];
  const oldAccount = createSolutionHistoryReader("alice", problem, (_, signal) => {
    requestCount++;
    requestSignal = signal;
    return new Promise((resolve) => { respond = resolve; });
  }, (state) => published.push(state));
  const inflight = oldAccount.loadMore();
  await oldAccount.loadMore();
  assert.equal(requestCount, 1, "Repeated clicks must not start overlapping history requests");
  oldAccount.dispose();
  assert.equal(requestSignal.aborted, true, "Account or problem changes must abort in-flight requests");
  const publishCount = published.length;
  respond(page([run(99)])); // A transport can still resolve after abort.
  await inflight;
  assert.equal(published.length, publishCount, "A late response must not publish a previous account's source");
  assert.equal(oldAccount.getState().runs.length, 0);

  const changedSession = createSolutionHistoryReader("alice", problem, async () => page([run(7)], "bob"), () => {});
  await changedSession.loadMore();
  assert.equal(changedSession.getState().error, true);
  assert.deepEqual(changedSession.getState().runs, [], "A changed server session must not display another user's submission");
  let currentUser = "alice";
  const laterSwitch = createSolutionHistoryReader("alice", problem, async () => page(firstPage, currentUser), () => {});
  await laterSwitch.loadMore();
  assert.equal(laterSwitch.getState().runs.length, 30);
  currentUser = "bob";
  await laterSwitch.loadMore();
  assert.equal(laterSwitch.getState().error, true);
  assert.deepEqual(laterSwitch.getState().runs, [], "A changed session must also clear previously loaded private records");
  let sessionExpired = false;
  const expiryReader = createSolutionHistoryReader("alice", problem, async () => {
    if (sessionExpired) throw new SolutionHistoryIdentityError("Authentication expired");
    return page(firstPage);
  }, () => {});
  await expiryReader.loadMore();
  sessionExpired = true;
  await expiryReader.loadMore();
  assert.deepEqual(expiryReader.getState().runs, [], "Expired access must clear previously loaded private records");
  console.log("PASS submission history exact source, pagination, retry, response identity and account-switch isolation");
} finally {
  await server.close();
}
