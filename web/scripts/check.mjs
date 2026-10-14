import assert from "node:assert/strict";
import fs from "node:fs";
import { createHash } from "node:crypto";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const { default: Lab } = await server.ssrLoadModule("/src/Lab.tsx");
  const { ArtifactCharts, Chart } = await server.ssrLoadModule("/src/Visuals.tsx");
  const { experimentTests } = await server.ssrLoadModule("/src/experimentResults.ts");
  const { isPassed, status } = await server.ssrLoadModule("/src/types.ts");
  const { parseReport, parseSaved, decodeProblemHash } =
    await server.ssrLoadModule("/src/reports.ts");
  const { fingerprintSource, reportSourceState } = await server.ssrLoadModule("/src/reportSource.ts");
  const { ReportSourceNotice } = await server.ssrLoadModule("/src/ReportSourceNotice.tsx");
  const { parseComparison } = await server.ssrLoadModule("/src/comparisons.ts");
  const { default: TestComparison } = await server.ssrLoadModule("/src/TestComparison.tsx");
  const { decodeViz } = await server.ssrLoadModule("/src/vizData.ts");
  const { highlight } = await server.ssrLoadModule("/src/highlight.ts");
  const { dpoObjective, tokenGradient, difficultyAdvantage, scheduleData } =
    await server.ssrLoadModule("/src/concepts.ts");
  for (const smoothing of [0, 0.1, 0.5]) {
    for (const margin of [-10000, -2, 0, 2, 10000]) {
      const result = dpoObjective(margin, 0.7, smoothing);
      assert(Number.isFinite(result.loss) && Number.isFinite(result.gradient));
      const step = 1e-3;
      const numerical =
        (dpoObjective(margin + step, 0.7, smoothing).loss -
          dpoObjective(margin - step, 0.7, smoothing).loss) /
        (2 * step);
      assert(
        Math.abs(numerical - result.gradient) < 1e-6,
        "DPO gradient must match loss derivative",
      );
    }
  }
  assert(
    dpoObjective(10000, 1, 0.1).gradient > 0,
    "Smoothing pulls extreme positive margins back",
  );
  const lengths = [3, 6, 12, 24];
  assert.equal(tokenGradient("GRPO", 1, 3, lengths), -1 / 12);
  assert.equal(tokenGradient("GRPO", 1, 24, lengths), -1 / 96);
  for (const length of lengths) {
    assert.equal(tokenGradient("DAPO", 1, length, lengths), -1 / 45);
    assert.equal(tokenGradient("Dr. GRPO", 1, length, lengths), -1 / 96);
  }
  assert.equal(difficultyAdvantage(0.5, false).grpo, 1);
  assert.equal(difficultyAdvantage(0.5, true).grpo, Math.sqrt(7 / 8));
  assert.equal(difficultyAdvantage(0.25).dr, 0.75);
  for (const count of [50, 200, 1000])
    for (const kind of ["linear", "cosine"])
      for (const offset of [0, 0.008, 0.05]) {
        const schedule = scheduleData(kind, count, offset);
        assert.equal(schedule.betas.length, count);
        let cumulative = 1;
        schedule.betas.forEach((beta, index) => {
          assert(beta > 0 && beta < 1);
          cumulative *= 1 - beta;
          assert.equal(schedule.bars[index], cumulative);
        });
        assert(schedule.bars.every(Number.isFinite));
      }
  const catalog = JSON.parse(fs.readFileSync("public/data/catalog.json", "utf8"));
  const { paperCollections, papersInCollection, collectionIdFromRoute } = await server.ssrLoadModule("/src/paperCollections.ts");
  assert.equal(collectionIdFromRoute("collections/llm-basics"), "llm-basics");
  assert.equal(collectionIdFromRoute("lists/llm-basics"), "llm-basics", "Keep shared legacy URLs usable");
  assert.equal(collectionIdFromRoute("papers/transformer"), undefined);
  assert.equal(collectionIdFromRoute("54"), undefined);
  const catalogPaperIds = new Set(catalog.papers.map((paper) => paper.id));
  assert.equal(new Set(paperCollections.map((list) => list.id)).size, paperCollections.length,
    "Collections need unique route IDs");
  for (const list of paperCollections) {
    assert(list.name && list.name_en && list.description && list.description_en && list.paperIds.length,
      `${list.id}: collections need bilingual names and papers`);
    assert.equal(new Set(list.paperIds).size, list.paperIds.length,
      `${list.id}: a paper must not be counted twice within one collection`);
    for (const id of list.paperIds)
      assert(catalogPaperIds.has(id), `${list.id}: unknown catalog paper ${id}`);
    const selected = papersInCollection([...catalog.papers].reverse(), list);
    assert.deepEqual(selected.map((paper) => paper.id), list.paperIds,
      `${list.id}: curated order must survive a different catalog order`);
  }
  assert.equal(papersInCollection(catalog.papers), catalog.papers,
    "All papers must preserve the catalog and include papers outside curated collections");
  const diffusionList = paperCollections.find((list) => list.id === "diffusion");
  const visualList = paperCollections.find((list) => list.id === "visual-generation");
  assert(visualList.paperIds.includes("var") && !diffusionList.paperIds.includes("var"),
    "VAR belongs to visual generation rather than the diffusion collection");
  assert(!diffusionList.paperIds.includes("drifting"),
    "Drifting Models should not be mislabeled as diffusion");
  const { PaperLibrary, matchingPaperModules } = await server.ssrLoadModule("/src/PaperViews.tsx");
  const transformer = catalog.papers.find((paper) => paper.id === "transformer");
  const matchingIds = (query, difficulty = "all") =>
    matchingPaperModules(transformer, catalog.problems, query, difficulty).map((problem) => problem.id);
  for (const query of ["正弦位置编码", "Sinusoidal Positional Encoding", "  SINUSOIDAL POSITIONAL ENCODING  "]) {
    assert.deepEqual(matchingIds(query), ["architecture/positional_encoding"],
      "A module-only search must not put unrelated siblings into the random pool");
    assert.deepEqual(matchingIds(query, "E"), ["architecture/positional_encoding"]);
    assert.deepEqual(matchingIds(query, "M"), [],
      "Search and difficulty must match the same module, not separate siblings");
  }
  for (const query of ["Transformer", "Attention Is All You Need", " "])
    assert.deepEqual(matchingIds(query), transformer.module_ids,
      "Matching the paper itself should retain all its modules");
  assert.deepEqual(matchingIds("Transformer", "M"), ["architecture/multihead_attention"],
    "A paper-level search must still honor the selected module difficulty");
  assert.deepEqual(matchingIds("absent module query"), []);
  const browserModules = catalog.problems.filter((p) => p.browser_judge);
  assert.equal(browserModules.length, 44);
  for (const id of ['representation/rotation_trick', 'world_models/lewm_prediction', 'world_models/lewm_planning', 'world_models/causal_jepa', 'representation/repa', 'representation/ijepa', 'finetuning/distillm2', 'finetuning/opsd_alignment', 'finetuning/opsd_clipping', 'rl/sdpo_feedback', 'rl/sdpo_topk', 'optimization/muon_split', 'finetuning/svdquant', 'architecture/flash_attention', 'architecture/kimi_kda', 'rl/glm5_offpolicy', 'rl/opd', 'diffusion/consistency', 'diffusion/progressive_distillation', 'diffusion/ddpm_sampling', 'diffusion/ddim', 'diffusion/flow_matching']) {
    const paper = catalog.papers.find(p => p.module_ids.includes(id));
    assert(matchingPaperModules(paper, catalog.problems, '', 'all', 'browser').some(p => p.id === id));
    assert(!matchingPaperModules(paper, catalog.problems, '', 'all', 'server').some(p => p.id === id));
  }
  const runtimeMatching = (query, runtime, difficulty = "all") => matchingPaperModules(transformer, catalog.problems, query, difficulty, runtime).map((p) => p.id);
  assert.deepEqual(runtimeMatching("Transformer", "browser"), transformer.module_ids);
  assert.deepEqual(runtimeMatching("Transformer", "server"), []);
  assert.deepEqual(runtimeMatching("LayerNorm", "browser"), ["architecture/layer_norm"]);
  assert.deepEqual(runtimeMatching("LayerNorm", "server"), []);
  assert.deepEqual(runtimeMatching("Transformer", "browser", "M"), ["architecture/multihead_attention"], "Difficulty and runtime must match the same module, including the random pool");
  const cachedStarter = catalog.problems.find((problem) => problem.id === "architecture/positional_encoding");
  assert.equal(status(cachedStarter, { code: cachedStarter.starter }), "todo",
    "A pristine cached starter must agree with random eligibility and the Not started filter");
  assert.equal(status(cachedStarter, { code: `${cachedStarter.starter}\n# Working draft` }), "started");
  assert.equal(status(cachedStarter, { code: "" }), "started", "An emptied editor is still an edited draft");
  assert.equal(status(cachedStarter, {
    code: cachedStarter.starter,
    report: { problem: cachedStarter.id, summary: { T0: [0, 1] }, tests: [{ id: "failed", layer: "T0", passed: false }] },
  }), "started", "Resetting a failed attempt to its starter does not make it untouched");
  console.log("PASS bilingual module searches, paper searches, difficulty conjunction, and pristine-starter status");
  const rowIds = (html) => [...html.matchAll(/href="#\/papers\/([^"]+)"/g)].map((match) => match[1]);
  const headingText = (html) => html.match(/<header\b[^>]*>([\s\S]*?)<\/header>/)[1]
    .replace(/<!--[\s\S]*?-->/g, "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ");
  const acceptedFixture = (module) => ({
    acceptedReport: {
      problem: module.id,
      summary: Object.fromEntries(Object.entries(module.tests).map(([layer, count]) => [layer, [count, count]])),
      tests: Object.entries(module.tests).flatMap(([layer, count]) =>
        Array.from({length: count}, (_, index) => ({id: `${layer}-${index}`, layer, passed: true}))),
    },
  });
  const libraryProps = {
    papers: catalog.papers, problems: catalog.problems, categories: catalog.categories, onGuide: () => {},
  };
  for (const list of paperCollections) {
    const members = catalog.papers.filter((paper) => list.paperIds.includes(paper.id));
    const moduleIds = new Set(members.flatMap((paper) => paper.module_ids));
    const inside = catalog.problems.find((module) => moduleIds.has(module.id));
    const outside = catalog.problems.find((module) => !moduleIds.has(module.id));
    const saved = {[inside.id]: acceptedFixture(inside), [outside.id]: acceptedFixture(outside)};
    const html = renderToString(React.createElement(PaperLibrary, {...libraryProps, saved, collectionId: list.id}));
    assert(html.includes(`href="#/collections/${list.id}" aria-current="page"`),
      `${list.id}: selected card must expose its canonical collection route`);
    assert(html.includes(`从当前范围内 ${moduleIds.size - 1} 道未开始的题目中随机选择`),
      `${list.id}: random candidates must exclude accepted modules and stay inside the collection`);
    const completed = Object.fromEntries(catalog.problems.filter((module) => moduleIds.has(module.id))
      .map((module) => [module.id, acceptedFixture(module)]));
    const completedHtml = renderToString(React.createElement(PaperLibrary,
      {...libraryProps, saved: completed, collectionId: list.id}));
    assert(/class="random-problem"[^>]*disabled/.test(completedHtml),
      `${list.id}: do not fall back to unsolved modules outside a fully completed collection`);
    assert.deepEqual(rowIds(html), list.paperIds,
      `${list.id}: the library must show exactly the selected papers in curated order`);
    assert(headingText(html).includes(`${list.name} ${members.length}`),
      `${list.id}: the heading must count papers within the selected collection`);
    assert(headingText(html).includes(`已通过模块 1 / ${moduleIds.size}`),
      `${list.id}: an accepted module outside this collection must not affect its progress`);
    const allHtml = renderToString(React.createElement(PaperLibrary, {...libraryProps, saved}));
    assert.deepEqual(rowIds(allHtml), catalog.papers.map((paper) => paper.id),
      "All papers must retain the complete catalog order");
    assert(headingText(allHtml).includes(`已通过模块 2 / ${catalog.problems.length}`),
      "All papers must include progress from both modules");
  }
  const unknownListHtml = renderToString(React.createElement(PaperLibrary, {
    ...libraryProps, saved: {}, collectionId: "not-a-maintained-list",
  }));
  assert.deepEqual(rowIds(unknownListHtml), catalog.papers.map((paper) => paper.id),
    "An unknown collection URL should still expose the paper library");
  const loadingHtml = renderToString(React.createElement(PaperLibrary,
    {...libraryProps, saved: {}, progressReady: false}));
  assert(/class="random-problem"[^>]*disabled/.test(loadingHtml), "Wait for account progress before drawing");
  const readyHtml = renderToString(React.createElement(PaperLibrary, {...libraryProps, saved: {}}));
  assert(!/class="random-problem"[^>]*disabled/.test(readyHtml), "Untouched modules should enable random selection");
  assert.equal([...readyHtml.matchAll(/class="collection-card /g)].length, paperCollections.length);
  assert(!readyHtml.includes("Study lists") && !readyHtml.includes("library-sidebar"));
  console.log("PASS featured collection membership, ordering, selection, routing, random availability, and scoped progress");
  const maintained = catalog.problems.filter((p) => !p.id.startsWith("community/"));
  for (const problem of maintained) {
    const file = problem.id.replaceAll("/", "-") + ".json";
    assert(fs.existsSync(`public/data/${file}`), `${file}: run paperleet vizdata all`);
    const data = JSON.parse(fs.readFileSync(`public/data/${file}`, "utf8"));
    const html = renderToString(React.createElement(Lab, {problem: {...problem, vizData: data}}));
    assert(!html.includes("NaN"), `${file} renders invalid numeric values`);
    assert(html.includes(data.kind === "toy_experiment" ? "参考实验" : "INTERACTIVE LAB"), `${file} has no visualization`);
    if (data.kind === "toy_experiment") {
      assert(html.includes("我的运行结果"), `${file} must distinguish submitted code from reference data`);
      assert(data.provenance, `${file} must explain toy provenance`);
      for (const chart of data.charts || []) {
        const heading = renderToString(React.createElement("h4", null, String(chart.title || "")));
        assert(html.includes(heading), `${file}: named chart ${chart.title} did not render`);
      }
    }
    console.log(`PASS ${file}`);
  }
  const publishedToy = renderToString(React.createElement(Lab, {
    problem: {id: "community/toy", title: "Reviewed toy", vizData: null},
    report: {problem: "community/toy", tests: [{id: "test_behavior", layer: "T3", passed: true,
      artifacts: {experiment: {kind: "toy_experiment", loss_curve: [[0, 1], [1, .1]], metrics: {error: .1}}}}]},
  }));
  assert(publishedToy.includes("test_behavior") && publishedToy.includes("loss_curve"));
  assert(publishedToy.includes("我的运行结果") && !publishedToy.includes(">参考实验</button>"),
    "Published toys without static previews must show actual judged artifacts, not an empty reference view");

  const packed = Buffer.alloc(8);
  [1000, -2000, 32767, -32768].forEach((value, i) =>
    packed.writeInt16LE(value, i * 2),
  );
  assert.deepEqual(
    decodeViz({
      encoding: "i16-base64",
      shape: [1, 2, 2],
      scale: 1000,
      data: packed.toString("base64"),
    }),
    [
      [
        [1, -2],
        [32.767, -32.768],
      ],
    ],
  );

  const report = {
    problem: "example",
    summary: { T0: [1, 1], T1: [1, 1], T2: [1, 1], T3: [0, 0] },
    tests: [
      {
        id: "test_contract",
        nodeid: "tests.py::test_contract",
        layer: "T0",
        passed: true,
      },
      {
        id: "test_reference",
        nodeid: "tests.py::test_reference",
        layer: "T1",
        passed: true,
      },
      {
        id: "test_property",
        nodeid: "tests.py::test_property",
        layer: "T2",
        passed: true,
      },
    ],
  };
  const problem = { id: "example", tests: { T0: 1, T1: 1, T2: 1, T3: 0 } };
  const comparison = {kind:'value',expected_shape:[2,3],actual_shape:[2,3],index:[1,2],expected:.5,actual:.25,absolute_error:.25,rtol:1e-5,atol:1e-8};
  assert.deepEqual(parseComparison(comparison),comparison);
  const comparisonHtml = renderToString(React.createElement(TestComparison,{comparison}));
  assert(comparisonHtml.includes('[1, 2]') && comparisonHtml.includes('0.5') && comparisonHtml.includes('0.25') && comparisonHtml.includes('允许误差'));
  const shapeComparison = {kind:'shape',expected_shape:[2,3],actual_shape:[3,2]};
  assert.deepEqual(parseComparison(shapeComparison),shapeComparison);
  assert(renderToString(React.createElement(TestComparison,{comparison:shapeComparison})).includes('张量形状不匹配'));
  const nonfiniteComparison = {...comparison,index:[],expected_shape:[],actual_shape:[],expected:'NaN',actual:'Infinity',absolute_error:'Infinity'};
  assert.deepEqual(parseComparison(nonfiniteComparison),nonfiniteComparison);
  assert(renderToString(React.createElement(TestComparison,{comparison:nonfiniteComparison})).includes('NaN 不会通过'));
  const exactInteger = {...comparison,actual:'1152921504606846976'};
  assert(renderToString(React.createElement(TestComparison,{comparison:exactInteger})).includes('1152921504606846976'));
  const failedComparisonReport = {...report,tests:[{...report.tests[0],passed:false,comparison},...report.tests.slice(1)],summary:{...report.summary,T0:[0,1]}};
  assert.deepEqual(parseReport(failedComparisonReport).tests[0].comparison,comparison);
  assert.equal(parseReport({...report,tests:report.tests.map(test=>({...test,comparison}))}).tests[0].comparison,undefined,'A passed row must not show stale failure diagnostics');
  for (const invalid of [null,[],{}, {...comparison,kind:'invented'}, {...comparison,index:[2,0]}, {...comparison,index:[true,1]}, {...comparison,rtol:-1}, {...comparison,atol:Infinity}, {...comparison,absolute_error:-1}, {...comparison,actual:'<script>alert(1)</script>'}, {...comparison,expected_shape:[2,3,4]}, {...comparison,expected_shape:Array(33).fill(1)}, {...shapeComparison,actual_shape:[2,3]}]) {
    assert.equal(parseComparison(invalid),undefined);
    assert.equal(parseReport({...failedComparisonReport,tests:[{...failedComparisonReport.tests[0],comparison:invalid},...failedComparisonReport.tests.slice(1)]}).tests[0].comparison,undefined,'Malformed optional diagnostics must leave the failed test and trace usable');
  }
  const source = '# 中文 🧪\nprint("hello")\n';
  const fingerprint = await fingerprintSource(source);
  assert.equal(fingerprint.sha256, createHash('sha256').update(source, 'utf8').digest('hex'), 'Browser and Python must hash identical UTF-8 bytes');
  const versioned = parseReport({...report,source_sha256:fingerprint.sha256});
  assert.equal(versioned.source_sha256, fingerprint.sha256, 'Keep code provenance through response parsing');
  assert.equal(parseSaved({example:{code:source,report:versioned}}).example.report.source_sha256, fingerprint.sha256, 'Retain provenance across refresh');
  assert.equal(reportSourceState(undefined,source,fingerprint),'none');
  assert.equal(reportSourceState(report,source,fingerprint),'unknown', 'Legacy reports must not claim to match current code');
  assert.equal(reportSourceState(versioned,source),'checking');
  assert.equal(reportSourceState(versioned,source,fingerprint),'current');
  const edited = source + '# new edit\n';
  assert.equal(reportSourceState(versioned,edited,fingerprint),'checking', 'An old asynchronous digest must never match newer editor contents');
  assert.equal(reportSourceState(versioned,edited,await fingerprintSource(edited)),'changed');
  assert.equal(reportSourceState(versioned,source,{source}),'unknown', 'Unavailable hashing must not block judging or invent a match');
  assert.notEqual((await fingerprintSource(source.replaceAll('\n','\r\n'))).sha256,fingerprint.sha256, 'Line endings are part of the submitted source');
  assert.equal(status(problem,{code:edited,report:versioned,acceptedReport:versioned}),'done', 'Editing after acceptance must preserve historical completion');
  const staleNotice = renderToString(React.createElement(ReportSourceNotice,{state:'changed'}));
  assert(staleNotice.includes('代码已修改') && staleNotice.includes('role="status"'));
  assert(isPassed(problem, report));
  assert(!isPassed(problem, { ...report, error: "collection failed" }));
  assert(!isPassed(problem, { ...report, summary: { T0: [1, 1] } }));
  assert(!isPassed(problem, { ...report, mutation: {} }));
  assert(!isPassed(problem, { ...report, problem: "another-problem" }));
  assert(!isPassed(problem, { ...report, tests: report.tests.slice(0, 1) }));
  assert(
    !isPassed(problem, {
      ...report,
      summary: { ...report.summary, T2: [9, 9] },
    }),
  );
  const partial = {
    ...report,
    tests: report.tests.slice(0, 1),
    summary: { T0: [1, 1] },
  };
  assert.doesNotThrow(() => parseReport(partial));
  assert(!isPassed(problem, partial));
  const parameterized = {
    ...report,
    tests: [
      ...report.tests,
      {
        id: "test_property",
        nodeid: "tests.py::test_property[other]",
        layer: "T2",
        passed: true,
      },
    ],
    summary: { ...report.summary, T2: [2, 2] },
  };
  assert(isPassed(problem, parameterized));
  assert(
    isPassed(problem, {
      ...parameterized,
      tests: parameterized.tests.map(({ nodeid, ...test }) => test),
    }),
  );
  assert(
    !isPassed(problem, {
      ...parameterized,
      tests: [...report.tests, report.tests[2]],
    }),
  );
  const badRows = [
    { duration_s: "bad" },
    { duration_s: -1 },
    { duration_s: Infinity },
    { duration_s: NaN },
    { why: {} },
    { message: [] },
    { id: null },
    { layer: "T99" },
    { passed: 1 },
    { nodeid: {} },
    { artifacts: [] },
  ];
  for (const bad of badRows)
    assert.throws(() =>
      parseReport({
        ...report,
        tests: [{ ...report.tests[0], ...bad }, ...report.tests.slice(1)],
      }),
    );
  for (const bad of [
    { mutation: { x: null } },
    { mutation: { x: [4] } },
    { impl: {} },
    { created_at: [] },
    { error: {} },
    { source_sha256: 'short' },
    { source_sha256: 'z'.repeat(64) },
    { source_sha256: {} },
    { summary: { ...report.summary, T0: [0.5, 1] } },
    { summary: { ...report.summary, T0: [2, 1] } },
    { summary: { ...report.summary, T0: [Infinity, Infinity] } },
    { summary: { ...report.summary, T9: [0, 0] } },
  ])
    assert.throws(() => parseReport({ ...report, ...bad }));
  assert.equal(parseReport(report).tests[0].duration_s, 0);
  assert.doesNotThrow(() =>
    parseReport({ ...report, mutation: { example: ["test_contract"] } }),
  );
  const recovered = parseSaved({
    example: { code: "print(1)", report },
    corrupt: {
      code: "draft survives",
      report: { ...report, problem: "corrupt", mutation: { x: null } },
    },
    mismatch: { report },
    nullEntry: null,
    arrayEntry: [],
    badCode: { code: [] },
  });
  assert.deepEqual(Object.keys(recovered).sort(), ["corrupt", "example"]);
  assert.equal(recovered.corrupt.code, "draft survives");
  assert.equal(recovered.corrupt.report, undefined);
  assert.equal(status(problem, {...recovered.example, report: undefined}), "done",
    "Legacy success should survive a reset of the latest report");
  const acceptedThenFailed = parseSaved({example: {acceptedReport: report, report: {
    ...report, error: "Later run failed",
  }}});
  assert.equal(status(problem, acceptedThenFailed.example), "done", "A later failed attempt must not erase completion");
  const wrongAcceptance = parseSaved({other: {acceptedReport: report}});
  assert.equal(wrongAcceptance.other, undefined, "Completion must belong to the same module");
  assert.equal(status(problem, {acceptedReport: partial}), "todo", "A passing subset does not complete a module");
  const { paperProgress } = await server.ssrLoadModule("/src/PaperViews.tsx");
  const paper = {module_ids: ["example", "optional"]};
  const practice = {problems: [{...problem, module: {optional: false}},
    {...problem, id: "optional", module: {optional: true}}], saved: acceptedThenFailed};
  assert.deepEqual(paperProgress(paper, practice), {completed: 1, total: 1, state: "done"},
    "Optional extensions must not invalidate a paper's completed core modules");
  for (const invalid of [null, [], 4, "bad"])
    assert.deepEqual(parseSaved(invalid), {});
  assert.equal(decodeProblemHash("#/%E0%A4%A"), "");
  assert.equal(decodeProblemHash("#/%"), "");
  assert.equal(decodeProblemHash("#/rl%2Fgae"), "rl/gae");

  const output = renderToString(
    React.createElement(ArtifactCharts, {
      data: {
        samples: [[1, 2]],
        reference_samples: [[2, 3]],
        mmd: 0.03,
        threshold: 0.05,
        loss_curve: [
          [0, 5],
          [1, 4],
        ],
        training: [
          { step: 0, reward: 0 },
          { step: 1, reward: 1 },
        ],
      },
    }),
  );
  assert(output.includes("你的实现与参考对照"));
  assert(output.includes("60.0"));
  const gapArtifacts = renderToString(React.createElement(ArtifactCharts, { data: { training: [
    {step: 0, reward: .2, wrong_length: 4.16},
    {step: 1, reward: .7, wrong_length: null},
    {step: 2, reward: .98, wrong_length: 3.5},
  ] } }));
  assert(gapArtifacts.includes('reward / step') && gapArtifacts.includes('wrong_length / step'), 'Optional null metrics must not hide valid reward/length curves');
  assert(!gapArtifacts.includes('NaN') && !gapArtifacts.includes('<pre>'));
  const displayOnlyHtml = renderToString(React.createElement(ArtifactCharts, {data:{experiment:{display_only:true,training:[{step:0,reward:.2},{step:1,reward:.9}]}}}));
  assert(displayOnlyHtml.includes('不按单次趋势判分'), 'Display-only experiments must keep their grading scope visible');
  const gapChart = renderToString(React.createElement(Chart, { series: [[[0,1],[1,NaN],[2,3]]] }));
  assert.match(gapChart, /d="M[^\"]* M/, 'Missing observations must break the plotted line instead of becoming zero or interpolated evidence');
  const ppoModule = catalog.problems.find((entry) => entry.id === 'rl/ppo_clip');
  const ownReport = {problem: ppoModule.id, runtime:'browser-numpy-wasm', tests:[
    {id:'browser_run_marker',layer:'T3',passed:true,artifacts:{bandit:{training:[{step:0,reward:0},{step:1,reward:.99}],best_prob:.99,max_kl:.09}}},
    {id:'not_an_experiment',layer:'T2',passed:true,artifacts:{must_stay_hidden:100}},
  ]};
  assert.equal(experimentTests(ppoModule.id, ownReport).length, 1);
  assert.deepEqual(experimentTests('rl/gae', ownReport), []);
  assert.deepEqual(experimentTests(ppoModule.id, {...ownReport,tests:[{id:'empty',layer:'T3',passed:false,artifacts:{}}]}), []);
  const runHtml = renderToString(React.createElement(Lab, {problem:ppoModule,report:ownReport}));
  assert(runHtml.includes('browser_run_marker') && runHtml.includes('CPU / WASM'), 'Completed experiments must open actual submitted-run data by default');
  assert(!runHtml.includes('must_stay_hidden') && !runHtml.includes('同一个 batch，多轮更新后会发生什么'));
  for (const module of [ppoModule,catalog.problems.find((entry)=>entry.id==='finetuning/lora')]) {
    const staleExperiment = renderToString(React.createElement(Lab,{problem:module,report:{...ownReport,problem:module.id},sourceState:'changed',initialView:'run'}));
    assert(staleExperiment.includes('代码已修改') && staleExperiment.includes('browser_run_marker'), 'Both interactive and generic labs must label stale submitted-run charts without hiding their evidence');
  }
  const explicitConcept = renderToString(React.createElement(Lab, {problem:ppoModule,report:ownReport,initialView:'concept'}));
  assert(!explicitConcept.includes('browser_run_marker') && explicitConcept.includes('同一个 batch，多轮更新后会发生什么'), 'Formula entry points must still open the interactive reference view after a completed run');
  const foreignHtml = renderToString(React.createElement(Lab, {problem:ppoModule,report:{...ownReport,problem:'rl/gae'}}));
  assert(!foreignHtml.includes('browser_run_marker'), 'Reports must never bleed between modules');
  const failedHtml = renderToString(React.createElement(Lab, {problem:ppoModule,report:{...ownReport,tests:[{...ownReport.tests[0],passed:false}]}}));
  assert(failedHtml.includes('browser_run_marker') && failedHtml.includes('未通过'), 'Failed T3 artifacts remain useful feedback with their failed status visible');
  for (const data of [
    { curve: [{ step: 0, loss: 1 }, null] },
    {
      curve: [
        { step: 0, loss: 1 },
        { step: 1, loss: "bad" },
      ],
    },
    {
      curve: [
        { step: 0, loss: 1 },
        { step: 1, loss: Infinity },
      ],
    },
    { curve: [{}, {}] },
    { samples: [[1, 2], null] },
    { curve: [0, Infinity] },
    { mmd: Infinity, threshold: Infinity },
    { value: undefined },
  ]) {
    const fallback = renderToString(
      React.createElement(ArtifactCharts, { data }),
    );
    assert(!fallback.includes("NaN"));
    assert(
      fallback.includes("<pre>"),
      "Malformed artifact should remain readable as JSON",
    );
  }
  for (const lang of ["python", "bash", "json", "yaml", "latex", "text"]) {
    const highlighted = await highlight("x = 1", lang);
    assert(highlighted.includes("<pre"));
    assert(highlighted.includes("--shiki-light:"));
    assert(highlighted.includes("--shiki-dark:"));
    assert(highlighted.includes("--shiki-light-bg:"));
    assert(highlighted.includes("--shiki-dark-bg:"));
  }
  console.log(
    "PASS packed trajectories, completion rules, report overlay, MMD threshold, syntax highlighting",
  );
} finally {
  await server.close();
}
