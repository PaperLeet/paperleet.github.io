// Bind the reusable production artifact to the tested source and build settings.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const metadata = '.paperleet-build.json';
const keys = ['PAPERLEET_MAIN_SITE_ENABLED', 'VITE_PATH_ROUTING', 'VITE_PUBLIC_SITE',
  'VITE_APP_EDITION', 'VITE_API_ORIGIN', 'VITE_HOSTED_APP_AVAILABLE', 'VITE_HOSTED_APP_URL'];
function settings(env) { return Object.fromEntries(keys.map(key => [key, env[key] || ''])); }
function inventory(directory, prefix = '') {
  const files = {};
  for (const entry of fs.readdirSync(path.join(directory, prefix), {withFileTypes:true}).sort((a,b) => a.name.localeCompare(b.name))) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (name === metadata) continue;
    assert(!entry.isSymbolicLink(), `Symlink in production artifact: ${name}`);
    if (entry.isDirectory()) Object.assign(files, inventory(directory, name));
    else {
      assert(entry.isFile(), `Non-file in production artifact: ${name}`);
      files[name] = crypto.createHash('sha256').update(fs.readFileSync(path.join(directory, name))).digest('hex');
    }
  }
  return files;
}
function artifact(mode, directory, sha, env = process.env) {
  assert(/^[a-f0-9]{40}$/.test(sha), 'Expected a full source revision');
  assert(fs.statSync(path.join(directory, 'index.html')).isFile(), 'Missing production homepage');
  const filename = path.join(directory, metadata);
  if (mode === 'write') {
    fs.writeFileSync(path.join(directory, 'paperleet-version.json'), JSON.stringify({source_revision:sha, build_method:'tested-quality-artifact'}) + '\n');
    fs.writeFileSync(filename, JSON.stringify({version:1, sha, settings:settings(env), files:inventory(directory)}) + '\n');
  } else {
    assert.equal(mode, 'verify', 'Expected write or verify');
    const proof = JSON.parse(fs.readFileSync(filename, 'utf8'));
    assert.equal(proof.version, 1);
    assert.equal(proof.sha, sha, 'Artifact revision differs from the tested release');
    assert.deepEqual(proof.settings, settings(env), 'Deployment settings changed; run quality checks with the new settings');
    assert.deepEqual(proof.files, inventory(directory), 'Production artifact contents changed');
  }
}
module.exports = {artifact, keys};
if (require.main === module) {
  artifact(...process.argv.slice(2));
  console.log('PASS production artifact source, settings and complete file hashes');
}
