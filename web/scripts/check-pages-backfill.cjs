const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const backfill = require('./backfill-pages-assets.cjs');

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'paperleet-pages-backfill-'));
  try {
    execFileSync('python3', ['-c', `
import io, json, pathlib, sys, tarfile, zipfile
root = pathlib.Path(sys.argv[1])
for name, window in [('old', 1800), ('current', 604800)]:
    target = root / (name + '.tar')
    with tarfile.open(target, 'w') as archive:
        data = json.dumps({'version': 1, 'grace_seconds': window}).encode()
        info = tarfile.TarInfo('pages-assets-manifest.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(root / (name + '.zip'), 'w') as archive:
        archive.write(target, 'artifact.tar')
`, root]);
    const old = path.join(root, 'old.tar'), current = path.join(root, 'current.tar');
    assert.equal(backfill.retainedWindow(old), 1800);
    assert.equal(backfill.retainedWindow(current), backfill.WINDOW_SECONDS);
    assert.equal(backfill.retainedWindow(path.join(root, 'missing.tar')), 0);
    const logs = [], calls = [];
    const now = Date.parse('2026-10-15T12:00:00Z');
    const core = {info: value => logs.push(value), warning:value => logs.push(value)};
    const context = {repo:{owner:'PaperLeet', repo:'paperleet.github.io'}, runId:99};
    const artifact = id => ({id, name:'paperleet-pages-assets', expired:false, expires_at:'2026-11-01T00:00:00Z', size_in_bytes:1024});
    const zip = await fs.readFile(path.join(root, 'old.zip'));
    const github = {
      rest:{actions:{listWorkflowRuns:'runs', listWorkflowRunArtifacts:'artifacts', downloadArtifact:async input => {
        calls.push(input);
        if (input.artifact_id === 3) throw Object.assign(Error('Expired'), {status:404});
        return {data:zip};
      }}},
      paginate:async (method, input) => {
        calls.push({method, ...input});
        if (method === 'runs') {
          assert.equal(input.created, '>=2026-10-08T12:00:00.000Z');
          assert.equal(input.status, 'success');
          return [99, 5, 4, 3, 2, 1].map(id => ({id}));
        }
        if (input.run_id === 4) return [{...artifact(4), name:'github-pages'}];
        return [artifact(input.run_id)];
      },
    };
    const options = {github, context, core, previousRun:5, previousArchive:current, directory:path.join(root, 'downloaded'), now};
    assert.equal(await backfill(options), 0);
    assert.equal(calls.length, 0, 'Normal deployments must not repeatedly download history');
    assert.equal(await backfill({...options, previousArchive:old}), 2);
    assert.equal(backfill.retainedWindow(path.join(options.directory, '2/artifact.tar')), 1800);
    assert.equal(backfill.retainedWindow(path.join(options.directory, '1/artifact.tar')), 1800);
    assert(!calls.some(item => [99, 5].includes(item.run_id)), 'Do not redownload the current or already downloaded previous run');
    assert(!calls.some(item => item.artifact_id === 4), 'Never backfill full Python runtime bundles');
    assert(logs.some(value => value.includes('expired before download')));
    assert(logs.some(value => value.includes('No lightweight')));
    assert.deepEqual(await fs.readdir(path.join(options.directory, '2')), ['artifact.tar']);
    const boundedGithub = {...github, paginate:async (method, input) => method === 'runs' ? [{id:1}] : [{...artifact(1), size_in_bytes:300 * 1024 * 1024}]};
    assert.equal(await backfill({...options, github:boundedGithub, previousArchive:old}), 0);
    assert(logs.some(value => value.includes('download budget reached')));
    console.log('PASS Pages asset backfill: policy migration, seven-day query, bounded downloads, expired artifacts, safe archive extraction, and no repeated history downloads');
  } finally { await fs.rm(root, {recursive:true, force:true}); }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
