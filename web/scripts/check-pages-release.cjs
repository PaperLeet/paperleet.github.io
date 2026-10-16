const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const resolve = require('./pages-release-gate.cjs');
const sha = 'a'.repeat(40), old = 'b'.repeat(40);
const repository = 'PaperLeet/paperleet.github.io';
function fixture() {
  return {
    context: {repo: {owner: 'PaperLeet', repo: 'paperleet.github.io'}, eventName: 'workflow_run', ref: 'refs/heads/main', sha, payload: {workflow_run: {id: 40}}},
    branch: {commit: {sha}}, workflow: {id: 7},
    run: {id: 40, workflow_id: 7, head_repository: {full_name: repository}, head_branch: 'main', head_sha: sha, event: 'push', status: 'completed', conclusion: 'success'},
    jobs: ['native', 'browser-wasm', 'pilot'].map(name => ({name, status: 'completed', conclusion: 'success'})),
  };
}
function adapter(f) {
  const actions = {
    getWorkflow: async args => { assert.equal(args.workflow_id, 'ci.yml'); return {data: f.workflow}; },
    getWorkflowRun: async args => { assert.equal(args.run_id, 40); return {data: f.run}; },
    listWorkflowRuns: async args => { assert.equal(args.branch, 'main'); assert.equal(args.workflow_id, 7); return {data: {workflow_runs: f.runs || [f.run]}}; },
    listJobsForWorkflowRun: () => { throw new Error('Use pagination for required job checks'); },
  };
  return {context: f.context, github: {rest: {actions, repos: {getBranch: async args => { assert.equal(args.branch, 'main'); return {data: f.branch}; }}}, paginate: async (method, args) => {
    assert.equal(method, actions.listJobsForWorkflowRun); assert.equal(args.filter, 'latest'); assert.equal(args.run_id, 40); return f.jobs;
  }}};
}
async function denied(edit, pattern) {
  const f = fixture(); edit(f); await assert.rejects(() => resolve(adapter(f)), pattern);
}
(async () => {
  assert.deepEqual(await resolve(adapter(fixture())), {sha, runId: 40});
  const manual = fixture(); manual.context.eventName = 'workflow_dispatch';
  manual.runs = [{...manual.run, id: 99, head_sha: old}, {...manual.run, id: 98, event: 'pull_request'}, manual.run];
  assert.deepEqual(await resolve(adapter(manual)), {sha, runId: 40}, 'Manual dispatch finds the exact main quality run, ignoring unrelated heads and PRs');
  for (const [key, value] of [['conclusion', 'failure'], ['conclusion', 'cancelled'], ['conclusion', 'timed_out'], ['status', 'in_progress'], ['head_sha', old], ['head_branch', 'feature'], ['event', 'pull_request'], ['workflow_id', 8]]) {
    await denied(f => { f.run[key] = value; }, /successful complete quality/);
  }
  await denied(f => { f.run.head_repository.full_name = 'someone/fork'; }, /successful complete quality/);
  await denied(f => { delete f.run.head_repository; }, /successful complete quality/);
  await denied(f => { f.branch.commit.sha = old; }, /successful complete quality/);
  await denied(f => { f.context.repo.owner = 'fork'; }, /official repository/);
  await denied(f => { f.context.eventName = 'push'; }, /main quality run/);
  await denied(f => { f.context.eventName = 'workflow_dispatch'; f.context.ref = 'refs/heads/feature'; }, /main quality run/);
  await denied(f => { f.context.eventName = 'workflow_dispatch'; f.context.sha = old; }, /main quality run/);
  await denied(f => { f.context.eventName = 'workflow_dispatch'; f.runs = []; }, /complete main quality gate/);
  await denied(f => { f.context.payload.workflow_run.id = '40'; }, /complete main quality gate/);
  for (const name of ['native', 'browser-wasm', 'pilot']) {
    await denied(f => { f.jobs = f.jobs.filter(job => job.name !== name); }, /missing or unsuccessful/);
    for (const conclusion of ['skipped', 'failure', 'cancelled', null]) {
      await denied(f => { f.jobs.find(job => job.name === name).conclusion = conclusion; }, /missing or unsuccessful/);
    }
    await denied(f => { f.jobs.find(job => job.name === name).status = 'in_progress'; }, /missing or unsuccessful/);
    await denied(f => { f.jobs.push({...f.jobs.find(job => job.name === name)}); }, /missing or unsuccessful/);
  }
  const workflow = fs.readFileSync(path.join(__dirname, '../../.github/workflows/pages.yml'), 'utf8');
  assert.match(workflow, /workflow_run:\n\s+workflows: \[PaperLeet quality gate\]\n\s+types: \[completed\]\n\s+branches: \[main\]/);
  assert(!/^  push:/m.test(workflow), 'A main push cannot publish ahead of its quality gate');
  assert.match(workflow, /ref: \$\{\{ github\.event\.workflow_run\.head_sha \|\| github\.sha \}\}/);
  assert.match(workflow, /sha: \$\{\{ steps\.release\.outputs\.sha \}\}/);
  assert.match(workflow, /if: steps\.current\.outputs\.publish == 'true'/);
  const buildCondition = workflow.split('    if: >-\n')[1].split('\n    runs-on:')[0];
  const eligible = new Function('github', `return (${buildCondition});`);
  function event() {
    return {repository, event_name: 'workflow_run', ref: 'refs/heads/main', sha, event: {workflow_run: {...fixture().run}}};
  }
  assert.equal(eligible(event()), true);
  const manualEvent = event(); manualEvent.event_name = 'workflow_dispatch'; delete manualEvent.event.workflow_run;
  assert.equal(eligible(manualEvent), true);
  for (const change of [
    e => { e.event.workflow_run.conclusion = 'failure'; },
    e => { e.event.workflow_run.head_sha = old; },
    e => { e.event.workflow_run.event = 'pull_request'; },
    e => { e.event.workflow_run.head_branch = 'feature'; },
    e => { e.event.workflow_run.head_repository.full_name = 'someone/fork'; },
    e => { e.repository = 'someone/fork'; },
    e => { e.event_name = 'push'; },
    e => { e.event_name = 'workflow_dispatch'; e.ref = 'refs/heads/feature'; },
  ]) {
    const e = event(); change(e); assert.equal(eligible(e), false, 'Invalid events must not enter the build concurrency group or checkout their source');
  }
  // Execute the actual pre-deploy script with delayed main changes and reruns.
  const script = workflow.split('name: Recheck the tested revision before publishing')[1].split('script: |\n')[1].split('\n      - uses:')[0].replace(/^ {12}/gm, '');
  const execute = new (Object.getPrototypeOf(async function() {}).constructor)('github', 'context', 'core', 'process', script);
  async function beforeDeploy(edit) {
    const f = fixture(); edit(f); const outputs = {};
    await execute(adapter(f).github, f.context, {setOutput: (name, value) => { outputs[name] = value; }, info: () => {}}, {env: {RELEASE_SHA: sha, QUALITY_RUN: '40'}});
    return outputs.publish;
  }
  assert.equal(await beforeDeploy(() => {}), 'true');
  assert.equal(await beforeDeploy(f => { f.branch.commit.sha = old; }), 'false', 'A main change during the build cannot publish stale assets');
  assert.equal(await beforeDeploy(f => { f.run.conclusion = 'failure'; }), 'false', 'A later failed rerun cannot reuse an earlier success');
  assert.equal(await beforeDeploy(f => { f.run.status = 'in_progress'; }), 'false');
  assert.equal(await beforeDeploy(f => { f.run.head_sha = old; }), 'false');
  console.log('PASS Pages release gating: complete exact-main native/WASM/pilot success, authoritative latest attempts, first-party provenance, manual dispatch, missing/skipped/duplicate job rejection and pre-deploy stale/rerun guards');
})().catch(error => { console.error(error); process.exitCode = 1; });
