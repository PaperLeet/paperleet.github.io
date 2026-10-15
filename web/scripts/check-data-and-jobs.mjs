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
    assert.deepEqual(item.paper, { title: "" }, "Module-only paper metadata loads with the detail");
    assert.deepEqual(item.functions, [], "Full signatures must not be duplicated in the homepage index");
    assert(fs.existsSync(`public/data/${item.detail_url}`));
    const raw = fs.readFileSync(`public/data/${item.detail_url}`, "utf8");
    const full = JSON.parse(raw);
    assert(full.paper?.title && full.functions.length, "Every maintained detail retains its paper and signatures");
    assert.equal(item.starter, full.starter, "Progress and backup comparisons keep the exact starter");
    assert.deepEqual(await loadProblemDetail(item, "/all-modules/", async () => new Response(raw)),
      { ...full, detail_url: item.detail_url }, "All maintained exports satisfy the rendering contract");
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
  assert.deepEqual(loaded.paper, detail.paper);
  assert.deepEqual(loaded.functions, detail.functions);
  assert.equal(reads, 1);
  await assert.rejects(loadProblemDetail(summary, "/", async () => Response.json({ ...detail, id: "wrong" })));
  await assert.rejects(loadProblemDetail({ ...summary, detail_url: "https://untrusted.test/code" }, "/"));
  assert.equal(await loadProblemDetail(detail, "/", () => { throw Error("Legacy details need no fetch"); }), detail);

  const invalidDetails = [null, [], 7, "not a module", summary,
    { ...detail, module: null },
    { ...detail, module: { ...detail.module, required_functions: null } },
    { ...detail, module: { ...detail.module, provided_functions: [null] } },
    { ...detail, module: { ...detail.module, goal: { text: "invalid React child" } } },
    { ...detail, tests: { ...detail.tests, T0: 999 } },
    { ...detail, browser_judge: { ...detail.browser_judge, backend: "unknown" } },
    { ...detail, starter: "different starter" },
    { ...detail, paper: null },
    { ...detail, paper: { ...detail.paper, title: {} } },
    { ...detail, paper: { ...detail.paper, authors: [null] } },
    { ...detail, paper: { ...detail.paper, anchors: [null] } },
    { ...detail, functions: null },
    { ...detail, functions: [{ name: "invalid", signature: {} }] },
    { ...detail, translations: { en: { title: {}, readme: "", notes: "" } } },
    { ...detail, translations: { en: { ...detail.translations.en, readme: [] } } },
    { ...detail, translations: { en: { ...detail.translations.en, notes: {} } } },
    { ...detail, solution: {} }, { ...detail, vizData: [] }];
  for (const [index, invalid] of invalidDetails.entries()) {
    await assert.rejects(loadProblemDetail(summary, `/invalid-detail-${index}/`, async () => Response.json(invalid)),
      /Module details do not match the catalog/, "Invalid data must not reach the React workspace");
  }
  const reordered = { ...detail, module: Object.fromEntries(Object.entries(detail.module).reverse()),
    tests: Object.fromEntries(Object.entries(detail.tests).reverse()) };
  assert.deepEqual((await loadProblemDetail(summary, "/reordered/", async () => Response.json(reordered))).module,
    detail.module, "Equivalent JSON objects may use a different key order");
  const cacheModes = [];
  const recoveryResponses = [Response.json(invalidDetails[6]), new Response("not JSON"), Response.json(detail), Response.json(detail)];
  const recovering = async (_url, options) => { cacheModes.push(options.cache); return recoveryResponses.shift(); };
  await assert.rejects(loadProblemDetail(summary, "/recover-detail/", recovering));
  await assert.rejects(loadProblemDetail(summary, "/recover-detail/", recovering));
  assert.equal((await loadProblemDetail(summary, "/recover-detail/", recovering)).id, summary.id);
  await loadProblemDetail(summary, "/recover-detail/", recovering);
  assert.deepEqual(cacheModes, ["force-cache", "reload", "reload", "force-cache"],
    "Retry must bypass a cached bad HTTP 200; a valid recovery restores immutable caching");
  let attempts = 0;
  const abandoned = new AbortController(), reason = new Error("Changed module");
  abandoned.abort(reason);
  await assert.rejects(loadProblemDetail(summary, "/aborted-detail/", () => { attempts++; }, abandoned.signal), reason);
  assert.equal(attempts, 0);
  const late = new AbortController();
  await assert.rejects(loadProblemDetail(summary, "/late-abort/", async () => {
    late.abort(reason); return Response.json(detail);
  }, late.signal), reason);
  await loadProblemDetail(summary, "/late-abort/", async (_url, options) => {
    assert.equal(options.cache, "force-cache", "Cancellation must not mark healthy content as corrupt");
    return Response.json(detail);
  });

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
  console.log("PASS split catalog: all maintained details, malformed rendering data, retry cache recovery, cancellation, size budget, identity and base paths; async jobs: transient recovery, GET-only polling, status validation, private access and abort cleanup");
} finally {
  await server.close();
}
