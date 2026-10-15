/** Recover cached-tab chunks once when extending the Pages retention window. */
const fs = require('node:fs/promises');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const WINDOW_SECONDS = 7 * 24 * 60 * 60;
const MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024;

function retainedWindow(archive) {
  try {
    return Number(execFileSync('python3', ['-c', `
import json, sys, tarfile
with tarfile.open(sys.argv[1], 'r:*') as archive:
    member = next((item for item in archive if item.name.lstrip('./') == 'pages-assets-manifest.json'), None)
    if member is None or not member.isfile() or member.size > 4 * 1024 * 1024:
        print(0)
    else:
        print(json.load(archive.extractfile(member)).get('grace_seconds', 0))
`, archive], {encoding:'utf8', stdio:['ignore','pipe','ignore']}));
  } catch { return 0; }
}

function unpack(zip, output) {
  // Read only artifact.tar, never extract arbitrary paths from an artifact ZIP.
  execFileSync('python3', ['-c', `
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    matches = [item for item in archive.infolist() if item.filename == 'artifact.tar']
    if len(matches) != 1 or matches[0].file_size > 256 * 1024 * 1024:
        raise ValueError('Expected one bounded lightweight asset archive')
    with archive.open(matches[0]) as source, open(sys.argv[2], 'xb') as target:
        shutil.copyfileobj(source, target)
`, zip, output], {stdio:'pipe'});
}

module.exports = async function backfill({github, context, core, previousRun, previousArchive, directory, now = Date.now(), wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))}) {
  if (!previousRun || retainedWindow(previousArchive) >= WINDOW_SECONDS) {
    core.info('No historical backfill needed; the previous artifact already carries the retention window.');
    return 0;
  }
  const cutoff = new Date(now - WINDOW_SECONDS * 1000).toISOString();
  const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
    ...context.repo, workflow_id:'pages.yml', branch:'main', status:'success', created:`>=${cutoff}`, per_page:100,
  });
  let bytes = 0, count = 0;
  for (const run of runs) {
    if (String(run.id) === String(previousRun) || run.id === context.runId) continue;
    const artifacts = await github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
      ...context.repo, run_id:run.id, per_page:100,
    });
    // Historical full Pages bundles include Python runtimes. Only lightweight
    // static assets are needed; never multiply those runtime downloads.
    const artifact = artifacts.find(item => item.name === 'paperleet-pages-assets' && !item.expired && Date.parse(item.expires_at) > now);
    if (!artifact) { core.warning(`No lightweight asset archive is available for Pages run ${run.id}.`); continue; }
    if (bytes + artifact.size_in_bytes > MAX_DOWNLOAD_BYTES) {
      core.warning('Historical asset download budget reached; older tabs can use the guarded reload recovery.');
      break;
    }
    let response;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        response = await github.rest.actions.downloadArtifact({...context.repo, artifact_id:artifact.id, archive_format:'zip'});
        break;
      } catch (error) {
        const status = Number(error?.status) || 0;
        if (status === 404 || status === 410) {
          core.warning(`Pages archive ${artifact.id} expired before download.`);
          break;
        }
        const transient = !status || status === 429 || status >= 500 && status < 600;
        // Never print the exception: artifact storage errors can contain signed URLs.
        if (transient && attempt < 2) {
          core.info(`Temporary ${status ? `HTTP ${status}` : 'network'} failure downloading Pages archive ${artifact.id}; retry ${attempt + 1}/2.`);
          await wait((attempt + 1) * 1000);
          continue;
        }
        core.warning(`Historical Pages archive ${artifact.id} is unavailable (${status ? `HTTP ${status}` : 'network error'}, ${attempt + 1} attempts); continuing with available assets.`);
        break;
      }
    }
    // Historical recovery is optional. A storage outage must not block the
    // validated current release or discard historical archives already fetched.
    if (!response) continue;
    const data = Buffer.from(response.data);
    if (bytes + data.byteLength > MAX_DOWNLOAD_BYTES) { core.warning('Historical asset download budget reached.'); break; }
    const target = path.join(directory, String(run.id));
    await fs.mkdir(target, {recursive:true});
    const zip = path.join(target, 'archive.zip');
    await fs.writeFile(zip, data);
    unpack(zip, path.join(target, 'artifact.tar'));
    await fs.unlink(zip);
    bytes += data.byteLength;
    count += 1;
    core.info(`Recovered lightweight assets from Pages run ${run.id}.`);
  }
  core.info(`Historical backfill: ${count} archives, ${Math.ceil(bytes / 1024 / 1024)} MiB downloaded.`);
  return count;
};
module.exports.retainedWindow = retainedWindow;
module.exports.WINDOW_SECONDS = WINDOW_SECONDS;
