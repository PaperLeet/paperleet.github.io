// Publication is an explicit version-tag action; ordinary development is local.
const officialRepository = 'PaperLeet/paperleet.github.io';
const shaPattern = /^[0-9a-f]{40}$/;
async function resolveTagRelease({github, context}) {
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  if (repository !== officialRepository) throw new Error('Releases require the official repository.');
  const match = /^refs\/tags\/(v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)$/.exec(context.ref || '');
  if (context.eventName !== 'push' || !match || context.payload?.ref !== context.ref ||
      context.payload.created !== true || context.payload.deleted === true) {
    throw new Error('Publication requires a newly created version tag, such as v0.33.0.');
  }
  const tag = match[1];
  const {data: reference} = await github.rest.git.getRef({...context.repo, ref: `tags/${tag}`});
  let object = reference.object;
  const tagObject = object?.sha, seen = new Set();
  for (let depth = 0; object?.type === 'tag'; depth++) {
    if (depth >= 8 || !shaPattern.test(object.sha) || seen.has(object.sha)) throw new Error('Invalid annotated tag chain.');
    seen.add(object.sha);
    const {data: annotated} = await github.rest.git.getTag({...context.repo, tag_sha: object.sha});
    object = annotated.object;
  }
  if (object?.type !== 'commit' || !shaPattern.test(object.sha) || !shaPattern.test(tagObject)) throw new Error('The release tag must resolve to a commit.');
  const sha = object.sha; seen.add(sha);
  if (!seen.has(context.sha) || (context.payload.after && !seen.has(context.payload.after))) throw new Error('The tag moved or its event revision is inconsistent.');
  const {data: branch} = await github.rest.repos.getBranch({...context.repo, branch: 'main'});
  if (!shaPattern.test(branch.commit.sha)) throw new Error('Invalid main revision.');
  const {data: comparison} = await github.request('GET /repos/{owner}/{repo}/compare/{basehead}', {
    ...context.repo, basehead: `${sha}...${branch.commit.sha}`,
  });
  if (!['ahead', 'identical'].includes(comparison.status) || comparison.behind_by !== 0 ||
      (comparison.status === 'identical' && branch.commit.sha !== sha)) {
    throw new Error('Release commits must be contained in main.');
  }
  return {sha, tag, tagObject};
}
module.exports = resolveTagRelease;
module.exports.stillCurrent = async function stillCurrent({github, context, expected}) {
  try {
    const current = await resolveTagRelease({github, context});
    return ['sha', 'tag', 'tagObject'].every(key => current[key] === expected[key]);
  } catch { return false; }
};
