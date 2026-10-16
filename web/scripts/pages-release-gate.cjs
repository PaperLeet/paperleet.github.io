// Pages must build the same first-party main revision that passed the full gate.
const requiredJobs = ['native', 'browser-wasm', 'pilot'];
module.exports = async function resolvePagesRelease({github, context}) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  if (repository !== 'PaperLeet/paperleet.github.io') throw new Error('Pages releases are limited to the official repository.');
  const {data: branch} = await github.rest.repos.getBranch({...context.repo, branch: 'main'});
  const sha = branch.commit.sha;
  if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error('Invalid main revision.');
  const {data: workflow} = await github.rest.actions.getWorkflow({...context.repo, workflow_id: 'ci.yml'});
  let runId;
  if (context.eventName === 'workflow_run') {
    runId = context.payload.workflow_run?.id;
  } else if (context.eventName === 'workflow_dispatch' && context.ref === 'refs/heads/main' && context.sha === sha) {
    const {data} = await github.rest.actions.listWorkflowRuns({...context.repo, workflow_id: workflow.id, branch: 'main', per_page: 100});
    runId = data.workflow_runs.find(run => run.head_sha === sha && ['push', 'workflow_dispatch'].includes(run.event))?.id;
  } else {
    throw new Error('Pages releases require a main quality run or a manual main dispatch.');
  }
  if (!Number.isSafeInteger(runId) || runId < 1) throw new Error('Run the complete main quality gate before publishing Pages.');
  // Read the authoritative latest attempt; a cached completion payload is insufficient.
  const {data: run} = await github.rest.actions.getWorkflowRun({...context.repo, run_id: runId});
  if (run.workflow_id !== workflow.id || run.head_repository?.full_name !== repository ||
      run.head_branch !== 'main' || run.head_sha !== sha || context.sha !== sha ||
      !['push', 'workflow_dispatch'].includes(run.event) || run.status !== 'completed' || run.conclusion !== 'success') {
    throw new Error('Pages requires a successful complete quality run for the current first-party main revision.');
  }
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {...context.repo, run_id: runId, filter: 'latest', per_page: 100});
  for (const name of requiredJobs) {
    const matching = jobs.filter(job => job.name === name);
    if (matching.length !== 1 || matching[0].status !== 'completed' || matching[0].conclusion !== 'success') {
      throw new Error(`Pages quality job is missing or unsuccessful: ${name}`);
    }
  }
  return {sha, runId};
};
