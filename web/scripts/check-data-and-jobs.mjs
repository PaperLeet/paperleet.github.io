import assert from "node:assert/strict";
import fs from "node:fs";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { loadProblemDetail } = await server.ssrLoadModule("/src/catalogData.ts");
  const { parseJob, requestJob, watchJob, pollDelay } = await server.ssrLoadModule("/src/judgeJobs.ts");
  const index = JSON.parse(fs.readFileSync("public/data/index.json", "utf8"));
  const summary = index.problems[0];
  assert(fs.statSync("public/data/index.json").size < 256 * 1024, "The homepage index has a 256 KB raw budget");
  for (const item of index.problems) {
    assert(!item.readme && !item.notes && !item.solution && !item.vizData);
    assert(item.starter && item.tests && item.module);
    assert(fs.existsSync(`public/data/${item.detail_url}`));
  }
  const detail = JSON.parse(fs.readFileSync(`public/data/${summary.detail_url}`, "utf8"));
  const signal = new AbortController().signal;
  let reads = 0;
  const loaded = await loadProblemDetail(summary, "/preview/", async (url, options) => {
    reads += 1;
    assert.equal(url, `/preview/data/${summary.detail_url}`);
    assert.equal(options.cache, "force-cache");
    assert.equal(options.signal, signal);
    return Response.json(detail);
  }, signal);
  assert.equal(loaded.solution, detail.solution);
  assert.equal(loaded.readme, detail.readme);
  assert.equal(reads, 1);
  await assert.rejects(loadProblemDetail(summary, "/", async () => Response.json({ ...detail, id: "wrong" })));
  await assert.rejects(loadProblemDetail({ ...summary, detail_url: "https://untrusted.test/code" }, "/"));
  assert.equal(await loadProblemDetail(detail, "/", () => { throw Error("Legacy details need no fetch"); }), detail);

  const id = "a".repeat(32);
  const report = { problem: summary.id, summary: { T0: [1, 1] }, tests: [{ id: "one", layer: "T0", passed: true }] };
  const queued = { id, problem: summary.id, status: "queued", ahead: 2 };
  const events = [], connection = [], paths = [];
  const responses = [new TypeError("network unavailable"), queued,
    { ...queued, status: "running", ahead: 0 }, { ...queued, status: "succeeded", report, ahead: 0 }];
  const result = await watchJob(id, "b".repeat(32), signal, (job) => events.push(job.status), (ready) => connection.push(ready),
    async (path, options) => {
      paths.push(path);
      assert.equal(options.method, "GET", "Reconnection must only poll, never resubmit source");
      assert.equal(options.headers["X-PaperLeet-Client-ID"], "b".repeat(32));
      const item = responses.shift();
      if (item instanceof Error) throw item;
      return Response.json(item);
    }, async () => {});
  assert.equal(result.report.problem, summary.id);
  assert.deepEqual(events, ["queued", "running", "succeeded"]);
  assert.equal(connection[0], false);
  assert(paths.every((path) => path === `/api/judge/jobs/${id}`));
  await assert.rejects(watchJob(id, "client", signal, () => {}, () => {},
    async () => Response.json({ error: "private job" }, { status: 401 }), () => { throw Error("Do not retry unauthorized reads"); }));
  await assert.rejects(watchJob(id, "client", signal, () => {}, () => {},
    async () => Response.json({ ...queued, id: "c".repeat(32) }), () => { throw Error("Do not retry a mismatched job"); }));
  assert.throws(() => parseJob({ ...queued, status: "succeeded" }));
  assert.throws(() => parseJob({ ...queued, ahead: -1 }));
  assert.throws(() => parseJob({ ...queued, report: { ...report, problem: "wrong" } }));
  let posted = 0;
  await requestJob("", "local-client", signal, { id, problem: summary.id, code: "frozen source" }, async (url, options) => {
    posted += 1;
    assert.equal(options.method, "POST");
    assert.equal(JSON.parse(options.body).code, "frozen source");
    return Response.json(queued, { status: 202 });
  });
  assert.equal(posted, 1);
  const stop = new AbortController();
  const waiting = pollDelay(10000, stop.signal);
  stop.abort(new Error("account changed"));
  await assert.rejects(waiting, /account changed/);
  console.log("PASS split catalog: size budget, immutable module details, identity and base paths; async jobs: transient recovery, GET-only polling, status validation, private access and abort cleanup");
} finally {
  await server.close();
}
