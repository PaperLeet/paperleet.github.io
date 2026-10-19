const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {artifact} = require('./pages-build-metadata.cjs');
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'paperleet-artifact-'));
const sha = 'a'.repeat(40), env = {VITE_APP_EDITION:'hosted', VITE_API_ORIGIN:'https://app.paperleet.com'};
try {
  fs.mkdirSync(path.join(root, 'assets'));
  fs.writeFileSync(path.join(root, 'index.html'), '<script src="/assets/app.js"></script>');
  fs.writeFileSync(path.join(root, 'assets/app.js'), 'tested');
  artifact('write', root, sha, env);
  artifact('verify', root, sha, env);
  assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'paperleet-version.json'))).source_revision, sha);
  assert.throws(() => artifact('verify', root, 'b'.repeat(40), env), /revision/);
  assert.throws(() => artifact('verify', root, sha, {...env, VITE_API_ORIGIN:'https://other.invalid'}), /settings changed/);
  fs.writeFileSync(path.join(root, 'assets/app.js'), 'tampered');
  assert.throws(() => artifact('verify', root, sha, env), /contents changed/);
  fs.writeFileSync(path.join(root, 'assets/app.js'), 'tested');
  fs.writeFileSync(path.join(root, 'extra.js'), 'unexpected');
  assert.throws(() => artifact('verify', root, sha, env), /contents changed/);
  fs.unlinkSync(path.join(root, 'extra.js'));
  fs.unlinkSync(path.join(root, '.paperleet-build.json'));
  assert.throws(() => artifact('verify', root, sha, env), /ENOENT/);
  fs.symlinkSync(path.join(root, 'index.html'), path.join(root, 'assets/link.js'));
  assert.throws(() => artifact('write', root, sha, env), /Symlink/);
  console.log('PASS reusable Pages artifact rejects wrong revisions, changed settings, tampered/missing/extra files and symlinks');
} finally {
  fs.rmSync(root, {recursive:true, force:true});
}
