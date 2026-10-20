import assert from "node:assert/strict";
import fs from "node:fs";
import { gzipSync } from "node:zlib";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { loadProblemDetail } = await server.ssrLoadModule("/src/catalogData.ts");
  const {decodeCatalogIndex,loadCatalogIndex}=await server.ssrLoadModule('/src/catalogIndex.ts');
  const { parseJob, requestJob, watchJob, pollDelay } = await server.ssrLoadModule("/src/judgeJobs.ts");
  const indexSource=fs.readFileSync('public/data/index-v4.json','utf8');
  const latest=JSON.parse(indexSource);
  const compact=JSON.parse(fs.readFileSync('public/data/index-v3.json','utf8'));
  const index = await decodeCatalogIndex(latest);
  assert.deepEqual(await decodeCatalogIndex(compact),index,'Already-open v3 clients retain the identical catalog');
  const legacy=JSON.parse(fs.readFileSync('public/data/index.json','utf8'));
  assert.deepEqual(index,legacy,'The transport preserves every starter byte, paper field and progress comparison');
  const summary = index.problems[0];
  assert(Buffer.byteLength(indexSource) < 256 * 1024, "The homepage index has a 256 KB raw budget");
  assert(gzipSync(indexSource).length < 96 * 1024, "The homepage index has a 96 KB gzip budget");
  assert.deepEqual(await decodeCatalogIndex([]),{problems:[],papers:[]});assert.deepEqual(await decodeCatalogIndex({problems:legacy.problems}),{problems:legacy.problems,papers:[]},'Legacy module-only consumers keep their established model');
  const urls=[];
  assert.deepEqual(await loadCatalogIndex('/preview/',async(url,options)=>{urls.push(url);assert.equal(options.cache,'no-cache');return Response.json(latest);}),legacy);
  assert.deepEqual(urls,['/preview/data/index-v4.json'],'The actual homepage request uses the bounded compact manifest');
  const fallback=[];
  assert.deepEqual(await loadCatalogIndex('/old/',async url=>{fallback.push(url);return url.endsWith('index-v4.json')?new Response('',{status:404}):Response.json(legacy);}),legacy);
  assert.deepEqual(fallback,['/old/data/index-v4.json','/old/data/index.json']);
  const decompressor=globalThis.DecompressionStream;
  try{globalThis.DecompressionStream=undefined;const reads=[];assert.deepEqual(await loadCatalogIndex('/compat/',async url=>{reads.push(url);return Response.json(legacy);}),legacy);assert.deepEqual(reads,['/compat/data/index.json'],'Browsers without the native decoder retain their established source and progress behavior');}finally{globalThis.DecompressionStream=decompressor;}
  for(const invalid of [null,{}, {...compact,schema_version:999}, {...compact,starters:{...compact.starters,encoding:'unknown'}}, {...compact,starters:{...compact.starters,bytes:1}}, {...compact,starters:{...compact.starters,bytes:3*1024*1024}}, {...compact,starters:{...compact.starters,data:'not a gzip body'}}, {...compact,problems:[...compact.problems,compact.problems[0]]}, {...compact,problems:compact.problems.map((row,i)=>i?row:{...row,id:'wrong/module'})}, {...compact,problems:compact.problems.map((row,i)=>i?row:{...row,starter:'competing source'})}])await assert.rejects(decodeCatalogIndex(invalid),'Malformed or oversized source transports cannot initialize an editor');
  const wire=packed=>{const bytes=Buffer.from(JSON.stringify(packed));return {schema_version:4,payload:{encoding:'gzip-base64',bytes:bytes.length,data:gzipSync(bytes).toString('base64')}};};
  for(const malformed of [null,{}, {tree:{'1':[]},shapes:[[]]}, {tree:{'0':[]},shapes:[['x']]}, {tree:{'0':[1,2]},shapes:[['x','x']]}, {tree:{'wrong':[]},shapes:[[]]}, {tree:{'0':[],'1':[]},shapes:[[]]}, {tree:{'0':[]},shapes:Array.from({length:257},()=>[])}])await assert.rejects(decodeCatalogIndex(wire(malformed)));
  for(const payload of [{...latest.payload,bytes:1},{...latest.payload,bytes:3*1024*1024},{...latest.payload,encoding:'unknown'},{...latest.payload,data:'not gzip'}])await assert.rejects(decodeCatalogIndex({...latest,payload}));
  const nested=[];let cursor=nested;for(let i=0;i<35;i++){const next=[];cursor.push(next);cursor=next;}await assert.rejects(decodeCatalogIndex(wire({shapes:[],tree:nested})),'Transport nesting is bounded');
  await assert.rejects(loadCatalogIndex('/',async()=>new Response('',{status:500})),/Catalog unavailable/);
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
