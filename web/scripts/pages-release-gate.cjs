// Publish only a successful explicit release of current first-party main.
const requiredJobs = ['native', 'infrastructure', 'browser-wasm', 'pilot'];
const version = /^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?![\s\S])/;
const explicit = run => (run.event === 'workflow_dispatch' && run.head_branch === 'main') || (run.event === 'push' && version.test(run.head_branch || ''));
module.exports = async function resolvePagesRelease({github, context}) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  if (repository !== 'PaperLeet/paperleet.github.io') throw new Error('Pages releases are limited to the official repository.');
  const {data: branch} = await github.rest.repos.getBranch({...context.repo, branch: 'main'});
  const sha = branch.commit.sha;
  if (!/^[0-9a-f]{40}(?![\s\S])/.test(sha)) throw new Error('Invalid main revision.');
  const {data: workflow} = await github.rest.actions.getWorkflow({...context.repo, workflow_id: 'ci.yml'});
  let runId;
  if (context.eventName === 'workflow_run') {
    runId = context.payload.workflow_run?.id;
  } else if (context.eventName === 'workflow_dispatch' && context.ref === 'refs/heads/main' && context.sha === sha) {
    const {data} = await github.rest.actions.listWorkflowRuns({...context.repo, workflow_id: workflow.id, per_page: 100});
    runId = data.workflow_runs.find(run => run.head_sha === sha && explicit(run))?.id;
  } else {
    throw new Error('Pages releases require an explicit release quality run or a manual main dispatch.');
  }
  if (!Number.isSafeInteger(runId) || runId < 1) throw new Error('Run the complete explicit release quality gate before publishing Pages.');
  const {data: run} = await github.rest.actions.getWorkflowRun({...context.repo, run_id: runId});
  if (run.workflow_id !== workflow.id || run.head_repository?.full_name !== repository ||
      run.head_sha !== sha || context.sha !== sha || !explicit(run) ||
      run.status !== 'completed' || run.conclusion !== 'success') {
    throw new Error('Pages requires a successful complete explicit quality run for the current first-party main revision.');
  }
  if (run.event === 'push') {
    const {data: ref} = await github.rest.git.getRef({...context.repo, ref: `tags/${run.head_branch}`});
    let object = ref.object;
    for (let i = 0; object.type === 'tag' && i < 4; i++) {
      const {data} = await github.rest.git.getTag({...context.repo, tag_sha: object.sha});
      object = data.object;
    }
    if (object.type !== 'commit' || object.sha !== sha) throw new Error('Release tag was moved or deleted.');
  }
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {...context.repo, run_id: runId, filter: 'latest', per_page: 100});
  for (const name of requiredJobs) {
    const matching = jobs.filter(job => job.name === name);
    if (matching.length !== 1 || matching[0].status !== 'completed' || matching[0].conclusion !== 'success') {
      throw new Error(`Pages quality job is missing or unsuccessful: ${name}`);
    }
  }
  // Overall workflow success may contain a skipped deploy; inspect the actual step.
  const previous = await github.paginate(github.rest.actions.listWorkflowRuns, {...context.repo, workflow_id: 'pages.yml', status: 'success', per_page: 100});
  for (const published of previous.filter(item => item.id !== context.runId && item.head_sha === sha)) {
    const history = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {...context.repo, run_id: published.id, filter: 'latest', per_page: 100});
    if (history.some(job => job.name === 'deploy' && job.conclusion === 'success' &&
        job.steps?.some(step => step.name === 'Run actions/deploy-pages@v4' && step.conclusion === 'success'))) {
      return {sha, runId, publish: false};
    }
  }
  return {sha, runId};
};
