import assert from "node:assert/strict";
import fs from "node:fs";
import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
try {
  const { loadProblemDetail } = await server.ssrLoadModule("/src/catalogData.ts");
  const {decodeCatalogIndex,loadCatalogIndex}=await server.ssrLoadModule('/src/catalogIndex.ts');
  const { parseJob, requestJob, watchJob, pollDelay } = await server.ssrLoadModule("/src/judgeJobs.ts");
  const indexSource=fs.readFileSync('public/data/index-v6.json','utf8');
  const latest=JSON.parse(indexSource);
  const v5=JSON.parse(fs.readFileSync('public/data/index-v5.json','utf8'));
  const previous=JSON.parse(fs.readFileSync('public/data/index-v4.json','utf8'));
  const compact=JSON.parse(fs.readFileSync('public/data/index-v3.json','utf8'));
  assert.match(latest.starters.url,/^starters\/[a-f0-9]{64}\.json$/);
  const starterSource=fs.readFileSync('public/data/'+latest.starters.url,'utf8');
  assert.equal(latest.starters.bytes,Buffer.byteLength(starterSource));
  assert.equal(latest.starters.sha256,createHash('sha256').update(starterSource).digest('hex'));
  assert.equal(latest.starters.url,`starters/${latest.starters.sha256}.json`);
  const starterFetcher=async(url,options)=>{assert.equal(url,'/preview/data/'+latest.starters.url);assert.equal(options.cache,'force-cache');assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');return new Response(starterSource);};
  const index = await decodeCatalogIndex(latest,'/preview/',starterFetcher);
  assert.deepEqual(await decodeCatalogIndex(v5),index,'Already-open v5 clients retain the identical catalog');
  assert.deepEqual(await decodeCatalogIndex(previous),index,'Already-open v4 clients retain the identical catalog');
  assert.deepEqual(await decodeCatalogIndex(compact),index,'Already-open v3 clients retain the identical catalog');
  const legacy=JSON.parse(fs.readFileSync('public/data/index.json','utf8'));
  assert.deepEqual(index,legacy,'The transport preserves every starter byte, paper field and progress comparison');
  const summary = index.problems[0];
  assert(Buffer.byteLength(indexSource) < 256 * 1024, "The homepage index has a 256 KB raw budget");
  assert(gzipSync(indexSource).length < 96 * 1024, "The homepage index has a 96 KB gzip budget");
  assert(gzipSync(starterSource).length < 64 * 1024, 'The immutable starter pack has a separate 64 KiB gzip budget');
  assert(gzipSync(indexSource).length+gzipSync(starterSource).length < 160 * 1024, 'The complete cold bootstrap stays within 160 KiB gzip without dropping source or metadata');
  assert.deepEqual(await decodeCatalogIndex([]),{problems:[],papers:[]});assert.deepEqual(await decodeCatalogIndex({problems:legacy.problems}),{problems:legacy.problems,papers:[]},'Legacy module-only consumers keep their established model');
  const urls=[];
  assert.deepEqual(await loadCatalogIndex('/preview/',async(url,options)=>{urls.push(url);if(url.endsWith('index-v6.json')){assert.equal(options.cache,'no-cache');return Response.json(latest);}return starterFetcher(url,options);}),legacy);
  assert.deepEqual(urls,['/preview/data/index-v6.json','/preview/data/'+latest.starters.url],'The homepage fetches fresh metadata and its exact immutable starter pack');
  for(const status of [404,410]){
    const fallback=[];
    assert.deepEqual(await loadCatalogIndex('/old/',async url=>{fallback.push(url);return url.endsWith('index-v6.json')?new Response('',{status}):Response.json(v5);}),legacy);
    assert.deepEqual(fallback,['/old/data/index-v6.json','/old/data/index-v5.json'],'An absent v6 falls back to the retained v5 manifest');
    const older=[];
    assert.deepEqual(await loadCatalogIndex('/older/',async url=>{older.push(url);return url.endsWith('index-v4.json')?Response.json(previous):new Response('',{status});}),legacy);
    assert.deepEqual(older,['/older/data/index-v6.json','/older/data/index-v5.json','/older/data/index-v4.json'],'Missing newer manifests still reach v4');
  }
  const fallback=[];
  assert.deepEqual(await loadCatalogIndex('/oldest/',async url=>{fallback.push(url);return url.endsWith('index.json')?Response.json(legacy):new Response('',{status:url.endsWith('index-v5.json')?404:410});}),legacy);
  assert.deepEqual(fallback,['/oldest/data/index-v6.json','/oldest/data/index-v5.json','/oldest/data/index-v4.json','/oldest/data/index.json']);
  const unavailable=await createServer({root:process.cwd(),configFile:false,esbuild:{jsx:'automatic'},server:{watch:null,middlewareMode:true,hmr:false,ws:false},appType:'custom',optimizeDeps:{noDiscovery:true,include:[]},plugins:[{name:'missing-catalog-codec',enforce:'pre',resolveId(source){if(source==='./catalogTransport')return '\0missing-catalog-codec';},load(id){if(id==='\0missing-catalog-codec')return 'throw Error("Decoder asset unavailable");';}}]});
  try{
    const {loadCatalogIndex:loadWithMissingCodec}=await unavailable.ssrLoadModule('/src/catalogIndex.ts');const reads=[];
    assert.deepEqual(await loadWithMissingCodec('/cached/',async(url,options)=>{reads.push(url);assert.equal(options.cache,'no-cache');return Response.json(legacy);}),legacy);
    assert.deepEqual(reads,['/cached/data/index.json'],'A missing optional decoder chunk requests only the unchanged legacy catalog');
  }finally{await unavailable.close();}
  const decompressor=globalThis.DecompressionStream;
  try{globalThis.DecompressionStream=undefined;const reads=[];assert.deepEqual(await loadCatalogIndex('/compat/',async url=>{reads.push(url);return Response.json(legacy);}),legacy);assert.deepEqual(reads,['/compat/data/index.json'],'Browsers without the native decoder retain their established source and progress behavior');}finally{globalThis.DecompressionStream=decompressor;}
  for(const invalid of [null,{}, {...compact,schema_version:999}, {...compact,starters:{...compact.starters,encoding:'unknown'}}, {...compact,starters:{...compact.starters,bytes:1}}, {...compact,starters:{...compact.starters,bytes:3*1024*1024}}, {...compact,starters:{...compact.starters,data:'not a gzip body'}}, {...compact,problems:[...compact.problems,compact.problems[0]]}, {...compact,problems:compact.problems.map((row,i)=>i?row:{...row,id:'wrong/module'})}, {...compact,problems:compact.problems.map((row,i)=>i?row:{...row,starter:'competing source'})}])await assert.rejects(decodeCatalogIndex(invalid),'Malformed or oversized source transports cannot initialize an editor');
  const wire=(packed,version=4)=>{const bytes=Buffer.from(JSON.stringify(packed));return {schema_version:version,payload:{encoding:'gzip-base64',bytes:bytes.length,data:gzipSync(bytes).toString('base64')}};};
  for(const malformed of [null,{}, {tree:{'1':[]},shapes:[[]]}, {tree:{'0':[]},shapes:[['x']]}, {tree:{'0':[1,2]},shapes:[['x','x']]}, {tree:{'wrong':[]},shapes:[[]]}, {tree:{'0':[],'1':[]},shapes:[[]]}, {tree:{'0':[]},shapes:Array.from({length:257},()=>[])}])await assert.rejects(decodeCatalogIndex(wire(malformed)));
  for(const payload of [{...latest.payload,bytes:1},{...latest.payload,bytes:3*1024*1024},{...latest.payload,encoding:'unknown'},{...latest.payload,data:'not gzip'}])await assert.rejects(decodeCatalogIndex({...latest,payload}));
  const nested=[];let cursor=nested;for(let i=0;i<35;i++){const next=[];cursor.push(next);cursor=next;}await assert.rejects(decodeCatalogIndex(wire({shapes:[],tree:nested})),'Transport nesting is bounded');
  const packed5={shapes:[['schema_version','problems','papers'],['id','starter','title']],strings:['network_biology/degree_distribution',"print('αβ')\n",''],tree:{'0':[2,[{'1':[{'s':0},{'s':1},{'s':2}]}],[]]}};
  assert.deepEqual(await decodeCatalogIndex(wire(packed5,5)),{schema_version:2,problems:[{id:packed5.strings[0],starter:packed5.strings[1],title:''}],papers:[]},'String references preserve Unicode, empty strings and every source newline');
  for(const strings of [undefined,null,{},'not a table',[null],[1],[{}],Array(100001).fill('')])await assert.rejects(decodeCatalogIndex(wire({...packed5,strings},5)),/Invalid catalog index/,'Only a bounded string table is accepted');
  for(const reference of [-1,.5,'1',null,true,{},[],3,Number.MAX_SAFE_INTEGER+1]){
    const bad=structuredClone(packed5);bad.tree['0'][1][0]['1'][1]={'s':reference};
    await assert.rejects(decodeCatalogIndex(wire(bad,5)),/Invalid catalog index/,'String references must be in-range safe integers');
  }
  for(const value of ['unencoded source',{'s':1,'0':[]},{'s':1,extra:0}]){
    const bad=structuredClone(packed5);bad.tree['0'][1][0]['1'][1]=value;
    await assert.rejects(decodeCatalogIndex(wire(bad,5)),/Invalid catalog index/,'V5 strings have one unambiguous reference representation');
  }
  const amplified={shapes:[['schema_version','problems','papers','extra']],strings:['x'.repeat(800000)],tree:{'0':[2,[],[],[{'s':0},{'s':0}]]}};
  assert.equal((await decodeCatalogIndex(wire(amplified,5))).extra.length,2,'Repeated values below the expanded text limit remain lossless');
  amplified.tree['0'][3].push({'s':0});
  await assert.rejects(decodeCatalogIndex(wire(amplified,5)),/Invalid catalog index/,'A small dictionary cannot amplify beyond the expanded text limit');
  await assert.rejects(decodeCatalogIndex(wire({...amplified,strings:['x'],tree:{'0':[2,[],[],Array(100001).fill({'s':0})]}},5)),/Invalid catalog index/,'String references also count toward the bounded node limit');
  await assert.rejects(decodeCatalogIndex(wire({shapes:[],strings:[],tree:nested},5)),/Invalid catalog index/,'V5 nesting remains bounded');
  let failedReads=0;
  await assert.rejects(loadCatalogIndex('/',async()=>{failedReads++;return new Response('',{status:500});}),/Catalog unavailable/);assert.equal(failedReads,1,'Server failures must not be hidden by a fallback catalog');
  let corruptReads=0;
  await assert.rejects(loadCatalogIndex('/',async()=>{corruptReads++;return Response.json(wire({...packed5,strings:[]},5));}),/Invalid catalog index/);assert.equal(corruptReads,1,'A corrupt 200 response cannot silently downgrade the catalog');
  let oversizedReads=0;
  await assert.rejects(loadCatalogIndex('/',async()=>{oversizedReads++;return new Response('x'.repeat(256*1024+1));}));assert.equal(oversizedReads,1,'The primary metadata response is byte-bounded before parsing and cannot trigger a fallback');
  // V6 is a transport split only: Unicode and source newlines hydrate before
  // any caller receives the established schema-2 catalog model.
  const packed6=structuredClone(packed5);packed6.strings[1]='';
  const sourceId=packed5.strings[0],sourceText=packed5.strings[1];
  const starterWire=value=>wire(value).payload;
  const splitFixture=(sources,asset={schema_version:1,payload:starterWire(sources)})=>{
    const body=JSON.stringify(asset),sha256=createHash('sha256').update(body).digest('hex');
    return {body,metadata:{...wire(packed6,6),starters:{url:`starters/${sha256}.json`,sha256,bytes:Buffer.byteLength(body)}}};
  };
  const split=splitFixture({[sourceId]:sourceText});
  const hydrate=(fixture=split)=>decodeCatalogIndex(fixture.metadata,'/nested/',async(url,options)=>{
    assert.equal(url,'/nested/data/'+fixture.metadata.starters.url);assert.equal(options.cache,'force-cache');assert.equal(options.credentials,'omit');assert.equal(options.redirect,'error');return new Response(fixture.body);
  });
  assert.deepEqual(await hydrate(),await decodeCatalogIndex(wire(packed5,5)));
  for(const starters of [undefined,null,{},
    {...split.metadata.starters,url:'https://untrusted.test/source.json'},
    {...split.metadata.starters,url:'../'+split.metadata.starters.url},
    {...split.metadata.starters,url:'/data/'+split.metadata.starters.url},
    {...split.metadata.starters,url:split.metadata.starters.url+'?cache=1'},
    {...split.metadata.starters,url:'starters/'+'A'.repeat(64)+'.json'},
    {...split.metadata.starters,sha256:'0'.repeat(64)},
    {...split.metadata.starters,bytes:0},
    {...split.metadata.starters,bytes:1.5},
    {...split.metadata.starters,bytes:256*1024+1},
  ]){
    let requests=0;await assert.rejects(decodeCatalogIndex({...split.metadata,starters},'/',async()=>{requests++;return new Response(split.body);}));assert.equal(requests,0,'Unsafe starter descriptors must be rejected before requesting any source');
  }
  for(const sources of [null,[],{},'source map',{[sourceId]:0},{[sourceId]:null},{other:sourceText},{[sourceId]:sourceText,other:'unexpected source'}])await assert.rejects(hydrate(splitFixture(sources)),'Starter IDs must exactly match metadata and each source must be a string');
  for(const asset of [null,{},[],{schema_version:999,payload:starterWire({[sourceId]:sourceText})},
    {schema_version:1,payload:{...starterWire({[sourceId]:sourceText}),encoding:'unknown'}},
    {schema_version:1,payload:{...starterWire({[sourceId]:sourceText}),bytes:1}},
    {schema_version:1,payload:{...starterWire({[sourceId]:sourceText}),bytes:3*1024*1024}},
    {schema_version:1,payload:{encoding:'gzip-base64',bytes:1,data:'not gzip'}},
  ])await assert.rejects(hydrate(splitFixture(null,asset)),'Malformed starter packs cannot initialize editors');
  const conflicting=structuredClone(packed5);
  await assert.rejects(hydrate({...split,metadata:{...split.metadata,payload:wire(conflicting,6).payload}}),'Metadata cannot compete with the starter pack for source bytes');
  const duplicate=structuredClone(packed6);duplicate.tree['0'][1].push(duplicate.tree['0'][1][0]);
  await assert.rejects(hydrate({...split,metadata:{...split.metadata,payload:wire(duplicate,6).payload}}),'Duplicate metadata IDs cannot hydrate');
  await assert.rejects(hydrate({...split,body:split.body.replace('"schema_version":1','"schema_version":2')}),'Equal-length response tampering must fail its SHA-256 check');
  await assert.rejects(hydrate({...split,metadata:{...split.metadata,starters:{...split.metadata.starters,bytes:split.metadata.starters.bytes+1}}}),'Declared raw byte length must match the immutable response');
  await assert.rejects(hydrate({...split,body:'x'.repeat(256*1024+1)}),'Oversized raw responses cannot bypass the bounded asset descriptor');
  const expanded=structuredClone(packed6);expanded.strings[2]='t'.repeat(1100000);
  const amplifiedSource=splitFixture({[sourceId]:'s'.repeat(1100000)});
  await assert.rejects(hydrate({...amplifiedSource,metadata:{...amplifiedSource.metadata,payload:wire(expanded,6).payload}}),'Combined hydrated metadata and source text retains the 2 MiB expansion bound');
  for(const status of [404,410,500]){
    const reads=[];await assert.rejects(loadCatalogIndex('/missing-source/',async url=>{reads.push(url);return url.endsWith('index-v6.json')?Response.json(split.metadata):new Response('',{status});}));
    assert.deepEqual(reads,['/missing-source/data/index-v6.json','/missing-source/data/'+split.metadata.starters.url],'Missing or failed starter assets do not silently downgrade a valid v6 catalog');
  }
  const corruptSourceReads=[];
  await assert.rejects(loadCatalogIndex('/corrupt-source/',async url=>{corruptSourceReads.push(url);return url.endsWith('index-v6.json')?Response.json(split.metadata):new Response(split.body.replace('"schema_version":1','"schema_version":2'));}));
  assert.equal(corruptSourceReads.length,2,'A corrupt 200 starter response must not trigger a fallback catalog');
  console.log('PASS catalog v6: exact v6/v5/v4/v3/legacy equality, 96 KiB metadata / 64 KiB immutable source / 160 KiB cold-bootstrap budgets, content-hash and byte verification, credential-free cached source fetch, bounded complete hydration and unchanged malformed/fallback protections');
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
