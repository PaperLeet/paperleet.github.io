// Admit explicit releases only; preserve the existing current-main deploy contract.
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const version = /^refs\/tags\/v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?![\s\S])/;
function checkReleaseRequest({env = process.env, git = (...args) => execFileSync('git', args, {encoding:'utf8'}).trim()} = {}) {
  if (env.GITHUB_REPOSITORY !== 'PaperLeet/paperleet.github.io') throw Error('Only the official repository can release.');
  const manual = env.GITHUB_EVENT_NAME === 'workflow_dispatch' && env.GITHUB_REF === 'refs/heads/main';
  const tagged = env.GITHUB_EVENT_NAME === 'push' && version.test(env.GITHUB_REF || '');
  if (!manual && !tagged) throw Error('Use manual main dispatch or a vMAJOR.MINOR.PATCH tag.');
  if (!/^[a-f0-9]{40}(?![\s\S])/.test(env.GITHUB_SHA || '')) throw Error('Invalid event revision.');
  const sha = git('rev-parse', 'HEAD');
  if (git('rev-parse', `${env.GITHUB_SHA}^{commit}`) !== sha) throw Error('Checkout does not match the release request.');
  const main = git('ls-remote', 'origin', 'refs/heads/main').split(/\s/)[0];
  if (!/^[a-f0-9]{40}(?![\s\S])/.test(sha) || main !== sha) throw Error('Release must still be current main.');
  if (tagged) {
    const refs = new Map(git('ls-remote', 'origin', env.GITHUB_REF, `${env.GITHUB_REF}^{}`).split('\n').filter(Boolean).map(line => {
      const [object, ref] = line.split(/\s+/); return [ref, object];
    }));
    if ((refs.get(`${env.GITHUB_REF}^{}`) || refs.get(env.GITHUB_REF)) !== sha) throw Error('Release tag was moved or deleted.');
  }
  return sha;
}
module.exports = {checkReleaseRequest, version};
if (require.main === module) {
  const sha = checkReleaseRequest();
  if (process.env.GITHUB_OUTPUT) fs.appendFileSync(process.env.GITHUB_OUTPUT, `sha=${sha}\n`);
  console.log(`Explicit release request verified: ${sha}`);
}
