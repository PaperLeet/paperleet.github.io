import assert from "node:assert/strict";
import fs from "node:fs";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { recommendPractice, nextPractice } = await server.ssrLoadModule("/src/practice-recommendation.ts");
  const { PracticeStart, PracticeNextStep } = await server.ssrLoadModule("/src/PracticeJourney.tsx");
  const { problems, papers } = JSON.parse(fs.readFileSync("public/data/catalog.json", "utf8"));
  const attention = problems.find((p) => p.id === "architecture/attention");
  const gae = problems.find((p) => p.id === "rl/gae");
  const rmsnorm = problems.find(p => p.id === 'architecture/rms_norm');
  const position = problems.find(p => p.id === 'architecture/positional_encoding');
  assert.equal(recommendPractice([position,rmsnorm], {})?.id,rmsnorm.id,'The foundations path introduces RMSNorm after LayerNorm');
  const transformer = papers.find((p) => p.id === "transformer");
  const accepted = (p) => ({
    problem: p.id,
    summary: Object.fromEntries(Object.entries(p.tests).map(([layer, count]) => [layer, [count, count]])),
    tests: Object.entries(p.tests).flatMap(([layer, count]) => Array.from({ length: count }, (_, i) => ({ id: `${layer}-${i}`, layer, passed: true }))),
  });
  const rope = problems.find(p => p.id === 'architecture/rope');
  assert.equal(recommendPractice([rope,position], {})?.id,position.id,'Introduce sinusoidal positions before rotary positions');
  assert.equal(recommendPractice([rope,position], {[position.id]:{report:accepted(position)}})?.id,rope.id,'Offer RoPE after position foundations');
  const gqa = problems.find(p => p.id === 'architecture/gqa');
  const multihead = problems.find(p => p.id === 'architecture/multihead_attention');
  assert.equal(recommendPractice([gqa,multihead], {})?.id,multihead.id,'Introduce ordinary multi-head attention before KV sharing');
  assert.equal(recommendPractice([gqa,multihead], {[multihead.id]:{report:accepted(multihead)}})?.id,gqa.id,'Offer GQA after the multi-head prerequisite');
  const failed = (p) => ({ problem: p.id, summary: { T0: [0, 1] }, tests: [{ id: "test", layer: "T0", passed: false }] });
  const pristine = { [gae.id]: { code: gae.starter, updated: "2099-01-01" } };
  assert.equal(recommendPractice([...problems].reverse(), pristine)?.id, attention.id,
    "First practice must be deterministic and cached starters must not count as unfinished drafts");
  const saved = {
    [attention.id]: { acceptedReport: accepted(attention), report: failed(attention), updated: "2099-01-01" },
    [gae.id]: { code: "# incomplete draft", updated: "2026-10-14T02:00:00Z" },
    [problems[1].id]: { report: failed(problems[1]), updated: "2026-10-14T01:00:00Z" },
    unknown: { code: "ignore stale catalog entry", updated: "2099-01-01" },
  };
  const before = structuredClone({ problems, saved });
  assert.equal(recommendPractice(problems, saved)?.id, gae.id, "Resume the newest unfinished module and preserve earlier acceptance");
  assert.deepEqual({ problems, saved }, before, "Recommendations must not reorder the catalog or change drafts");
  assert.equal(recommendPractice([gae], {})?.id, gae.id, "A collection must not suggest modules outside its scope");
  assert.equal(recommendPractice([{ ...gae, module: { ...gae.module, optional: true } }], {}), undefined,
    "Optional extensions must not be mistaken for required core progress");
  assert.equal(nextPractice(attention, transformer.module_ids, problems, saved)?.id, transformer.module_ids[1],
    "After passing, continue the paper in declared order before resuming a different paper");
  const allAccepted = Object.fromEntries(problems.map((p) => [p.id, { report: accepted(p) }]));
  assert.equal(recommendPractice(problems, allAccepted), undefined);
  assert.equal(nextPractice(attention, transformer.module_ids, problems, allAccepted), undefined);
  assert.equal(recommendPractice([], {}), undefined);
  const transformerDone = Object.fromEntries(transformer.module_ids.map((id) => [id, allAccepted[id]]));
  assert.notEqual(nextPractice(attention, transformer.module_ids, problems, transformerDone)?.paper_id, "transformer",
    "A completed paper should lead to an unsolved module in another paper");
  const props = { problems, saved: {}, ready: true };
  const firstVisit = renderToString(React.createElement(PracticeStart, props));
  assert(firstVisit.includes(`href="#/${attention.number}"`) && firstVisit.includes("推荐第一题"));
  const waiting = renderToString(React.createElement(PracticeStart, { ...props, ready: false }));
  assert(!waiting.includes(`href="#/${attention.number}"`) && waiting.includes('aria-busy="true"'),
    "Do not offer a new-user recommendation before private progress has loaded");
  const complete = renderToString(React.createElement(PracticeStart, { ...props, saved: allAccepted }));
  assert(complete.includes("核心模块已完成") && !complete.includes("开始这道题"));
  const next = renderToString(React.createElement(PracticeNextStep, { next: gae, samePaper: false, onNotes: () => {}, onLab: () => {} }));
  assert(next.includes(`href="#/${gae.number}"`) && next.includes("复盘常见坑点") && next.includes("查看实验结果"));
  console.log("PASS practice journey: deterministic first module, scoped recommendations, recent drafts, accepted preservation, paper sequence, loading and completion states");
} finally {
  await server.close();
}
