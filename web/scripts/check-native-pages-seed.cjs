const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');

const workflow = fs.readFileSync(path.resolve(__dirname, '../../.github/workflows/pages.yml'), 'utf8');
const step = workflow.split('      - name: Snapshot branch-published assets\n')[1]
  ?.split('      - name: Retain assets for cached pages\n')[0];
assert(step, 'The native Pages retention snapshot step must exist');
const shell = step.split('        run: |\n')[1].split('\n')
  .filter(line => line.startsWith('          ')).map(line => line.slice(10)).join('\n');
assert(shell.includes('--filter=blob:none'), 'Only the selected asset blobs should download');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'paperleet-native-pages-seed-'));
const env = {...process.env, GIT_CONFIG_NOSYSTEM:'1', GIT_CONFIG_GLOBAL:'/dev/null'};
function command(program, args, options = {}) {
  const result = spawnSync(program, args, {encoding:'utf8', env, ...options});
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result.stdout;
}
try {
  const server = path.join(root, 'server');
  const bare = path.join(server, 'team', 'app.git');
  const work = path.join(root, 'work');
  fs.mkdirSync(path.dirname(bare), {recursive:true});
  command('git', ['init', '--bare', bare]);
  command('git', ['-C', bare, 'config', 'uploadpack.allowFilter', 'true']);
  command('git', ['init', '--initial-branch=main', work]);
  fs.mkdirSync(path.join(work, 'assets'));
  fs.writeFileSync(path.join(work, 'assets', 'cached.js'), 'cached-page dependency\n');
  fs.writeFileSync(path.join(work, 'index.html'), 'Old HTML must not be seeded');
  fs.writeFileSync(path.join(work, 'runtime.wasm'), 'Old runtime must not be seeded');
  fs.writeFileSync(path.join(work, 'pages-assets-manifest.json'), JSON.stringify({version:1,
    grace_seconds:604800, generations:[{created_at:'2026-10-19T00:00:00Z', assets:['assets/cached.js']}]}));
  command('git', ['-C', work, 'add', '.']);
  command('git', ['-C', work, '-c', 'user.name=Retention fixture', '-c',
    'user.email=fixture@example.invalid', 'commit', '-m', 'Published cache dependency']);
  command('git', ['-C', work, 'push', bare, 'HEAD:refs/heads/gh-pages']);
  function snapshot(repository, label) {
    const temp = path.join(root, label);
    fs.mkdirSync(temp);
    const result = spawnSync('bash', ['-euo', 'pipefail', '-c', shell], {encoding:'utf8', env:{...env,
      GITHUB_SERVER_URL:pathToFileURL(server).href, GITHUB_REPOSITORY:repository, RUNNER_TEMP:temp}});
    return {temp, result};
  }
  const present = snapshot('team/app', 'present');
  assert.equal(present.result.status, 0, present.result.stderr);
  const archive = path.join(present.temp, 'paperleet-native-assets.tar');
  const members = command('tar', ['-tf', archive]).trim().split('\n');
  assert.deepEqual(members.sort(), ['assets/', 'assets/cached.js', 'pages-assets-manifest.json']);
  assert.equal(command('tar', ['-xOf', archive, 'assets/cached.js']), 'cached-page dependency\n');
  const manifest = JSON.parse(command('tar', ['-xOf', archive, 'pages-assets-manifest.json']));
  assert.equal(manifest.grace_seconds, 604800);
  assert.deepEqual(manifest.generations[0].assets, ['assets/cached.js']);
  assert(!fs.existsSync(path.join(present.temp, 'paperleet-native-pages', 'index.html')));
  assert(!fs.existsSync(path.join(present.temp, 'paperleet-native-pages', 'runtime.wasm')));
  command('git', ['init', '--bare', path.join(server, 'team', 'empty.git')]);
  const absent = snapshot('team/empty', 'absent');
  assert.equal(absent.result.status, 0, absent.result.stderr);
  assert(!fs.existsSync(path.join(absent.temp, 'paperleet-native-assets.tar')));
  const broken = snapshot('team/missing', 'broken');
  assert.notEqual(broken.result.status, 0, 'A failed remote read must never become an empty-history success');
  console.log('PASS native Pages seed: branch-only cached chunks and manifest retained; HTML/runtime excluded; missing branch accepted and remote failures propagated.');
} finally {
  fs.rmSync(root, {recursive:true, force:true});
}
