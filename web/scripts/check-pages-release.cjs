const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const resolve = require('./pages-release-gate.cjs');
const sha = 'a'.repeat(40), old = 'b'.repeat(40);
const repository = 'PaperLeet/paperleet.github.io';
function fixture() {
  return {
    context: {repo: {owner: 'PaperLeet', repo: 'paperleet.github.io'}, eventName: 'workflow_run', ref: 'refs/heads/main', sha, payload: {workflow_run: {id: 40}}},
    branch: {commit: {sha}}, workflow: {id: 7},
    run: {id: 40, workflow_id: 7, head_repository: {full_name: repository}, head_branch: 'main', head_sha: sha, event: 'push', status: 'completed', conclusion: 'success'},
    jobs: ['native', 'infrastructure', 'browser-wasm', 'pilot'].map(name => ({name, status: 'completed', conclusion: 'success'})),
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
  for (const name of ['native', 'infrastructure', 'browser-wasm', 'pilot']) {
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
  const ci = fs.readFileSync(path.join(__dirname, '../../.github/workflows/ci.yml'), 'utf8');
  const infrastructure = ci.split('  infrastructure:\n')[1].split('  browser-wasm:\n')[0];
  assert.match(infrastructure, /run: python -m pytest tests packages\/stridetorch\/tests -q\n/);
  assert.match(ci.split('  native:\n')[1].split('  infrastructure:\n')[0], /run: paperleet ci --budget 600\n/);
  const wasm = ci.split('  browser-wasm:\n')[1].split('  pilot:\n')[0];
  assert.match(wasm, /PAPERLEET_WASM_CONCURRENCY: '2'/);
  assert.match(wasm, /npm run check:browser-wasm\n/);
  const pilot = ci.split('  pilot:\n')[1];
  assert.match(pilot, /needs: \[native, infrastructure, browser-wasm\]/);
  assert.match(pilot, /if: always\(\)/);
  for (const [key, expression] of [['NATIVE_RESULT','needs.native.result'], ['INFRASTRUCTURE_RESULT','needs.infrastructure.result'], ['WASM_RESULT',"needs['browser-wasm'].result"]]) {
    assert(pilot.includes(key + ': $' + '{{ ' + expression + ' }}'), 'Pilot results must come from their actual required jobs');
  }
  const shell = pilot.split('        run: |\n')[1].replace(/^ {10}/gm, '');
  const successes = {NATIVE_RESULT:'success', INFRASTRUCTURE_RESULT:'success', WASM_RESULT:'success'};
  const pilotStatus = values => spawnSync('sh', ['-eu', '-c', shell], {env:{...process.env,...values}, encoding:'utf8'}).status;
  assert.equal(pilotStatus(successes), 0);
  for (const key of Object.keys(successes)) {
    for (const status of ['failure', 'cancelled', 'skipped', 'timed_out', '']) {
      assert.notEqual(pilotStatus({...successes,[key]:status}), 0, 'Every missing or unsuccessful check must block the actual pilot shell');
    }
  }
  console.log('PASS Pages release gating: complete exact-main infrastructure/native/WASM/pilot success, authoritative attempts, provenance, missing/skipped/duplicate rejection, actual pilot failure propagation and stale/rerun guards');
})().catch(error => { console.error(error); process.exitCode = 1; });
