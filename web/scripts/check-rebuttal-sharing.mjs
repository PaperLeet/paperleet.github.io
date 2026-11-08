import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({ configFile: false, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{ name: 'sharing-api', enforce: 'pre', resolveId(source) { if (source === './social') return '\0sharing-api'; }, load(id) { if (id === '\0sharing-api') return 'export const socialApi=()=>{throw Error("Unexpected live request")};'; } }],
});
const stores = [];
const deferred = () => { let resolve; const promise = new Promise(yes => { resolve = yes; }); return { promise, resolve }; };
const problem = (status, code) => Object.assign(Error(code), { status, code });
try {
  const { createRebuttalShareLinks, createRebuttalJoin } = await server.ssrLoadModule('/src/rebuttalSharing.ts');
  const { makeRebuttalShareToken, isRebuttalShareToken, rebuttalShareHref, rememberRebuttalShareToken, readRebuttalShareToken, forgetRebuttalShareToken } = await server.ssrLoadModule('/src/rebuttalShareHandoff.ts');
  const { readRebuttalRoute, readRebuttalPaper, createRebuttalCloud } = await server.ssrLoadModule('/src/rebuttalCloud.ts');
  const { newProject } = await server.ssrLoadModule('/src/rebuttalModel.ts');
  const values = new Map(), storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const token = makeRebuttalShareToken();
  assert(isRebuttalShareToken(token)); assert.equal(token.length, 43); assert.notEqual(token, makeRebuttalShareToken());
  assert.deepEqual(readRebuttalRoute(rebuttalShareHref(token)), { join: true, token });
  assert.deepEqual(readRebuttalRoute('#/rebuttal/join'), { join: true });
  assert.deepEqual(readRebuttalRoute('#/rebuttal/join/not-a-secret'), { join: true, invalid: true });
  rememberRebuttalShareToken(storage, token, 1000); assert.equal(readRebuttalShareToken(storage, 1100), token);
  assert.throws(() => readRebuttalShareToken(storage, 1_802_000)); assert.equal(values.size, 0);
  rememberRebuttalShareToken(storage, token, 1000); forgetRebuttalShareToken(storage); assert.equal(readRebuttalShareToken(storage), undefined);
  assert.throws(() => rememberRebuttalShareToken({ ...storage, setItem() { throw Error('Storage blocked'); } }, token));
  values.set('paperleet.rebuttal.share-handoff.v1', 'corrupt'); assert.throws(() => readRebuttalShareToken(storage)); assert.equal(values.size, 0);

  const owner = { id: 'owner_a', username: 'owner_a', display_name: 'Owner A' }, paperId = 'paper_a';
  const blank = role => ({ role, revision: 0, active: false, id: null, created_at: null, expires_at: null });
  const live = (role, revision = 1) => ({ role, revision, active: true, id: `link_${role}_${revision}`, created_at: '2030-01-01T00:00:00Z', expires_at: '2030-01-31T00:00:00Z' });
  let links = [blank('viewer'), blank('editor')], lost = true;
  const requests = [], response = link => ({ user_id: owner.id, paper_id: paperId, link });
  const request = async (path, body) => {
    assert.equal(path, `rebuttal/papers/${paperId}/links`);
    if (!body) return { user_id: owner.id, paper_id: paperId, links };
    requests.push(structuredClone(body));
    assert.equal(body.expected_user_id, owner.id);
    const link = body.action === 'create' ? live(body.role, body.expected_revision + 1) : { ...links.find(item => item.role === body.role), active: false, revision: body.expected_revision + 1 };
    links = links.map(item => item.role === link.role ? link : item);
    if (lost) { lost = false; throw Error('Lost successful response'); }
    return response(link);
  };
  const createLinks = options => { const store = createRebuttalShareLinks({ owner: owner.id, paperId, request, ...options }); stores.push(store); return store; };
  const control = createLinks(); await control.load();
  await control.change('viewer', 'create'); assert.equal(control.getSnapshot().pending, true); assert.equal(control.getSnapshot().tokens.viewer, undefined);
  const original = structuredClone(requests[0]);
  await control.change('editor', 'create'); assert.equal(requests.length, 1, 'Competing operation cannot replace an uncertain token request');
  await control.change(); assert.deepEqual(requests[1], original); assert.equal(control.getSnapshot().tokens.viewer.token, original.token);
  await control.change('editor', 'create'); assert.equal(requests[2].expected_revision, 0, 'Roles have independent revisions');
  assert.notEqual(requests[2].token, original.token);
  await control.change('viewer', 'revoke'); assert.equal(control.getSnapshot().tokens.viewer, undefined); assert(!('token' in requests.at(-1)));
  const reloaded = createLinks(); await reloaded.load(); assert.deepEqual(reloaded.getSnapshot().tokens, {}, 'Server metadata cannot recover bearer tokens');
  links = links.map(item => item.role === 'editor' ? live('editor', 2) : item); await control.load(); assert.equal(control.getSnapshot().tokens.editor, undefined, 'External rotation invalidates cached copy');
  const mismatch = createLinks({ request: async () => ({ user_id: 'other', paper_id: paperId, links }) }); await mismatch.load(); assert.equal(mismatch.getSnapshot().ready, false);
  let account = owner.id; const delayed = deferred();
  const switched = createLinks({ canAct: () => account === owner.id, request: async (_path, body) => body ? delayed.promise : { user_id: owner.id, paper_id: paperId, links } });
  await switched.load(); const oldAction = switched.change('viewer', 'create'); account = 'other'; delayed.resolve(response(live('viewer', links[0].revision + 1)));
  assert.equal(await oldAction, undefined); assert.deepEqual(switched.getSnapshot().tokens, {});

  const preview = { user_id: 'recipient_a', link: { ...live('viewer'), paper_id: paperId, title: 'Private paper', venue: 'ICLR', owner }, member_role: 'editor' };
  const joinCalls = []; let acceptanceLost = true;
  const join = createRebuttalJoin({ owner: 'recipient_a', token, request: async (path, body) => {
    assert.equal(path, 'rebuttal/join'); joinCalls.push(structuredClone(body));
    if (body.action === 'preview') return preview;
    if (acceptanceLost) { acceptanceLost = false; throw Error('Connection lost'); }
    return { user_id: 'recipient_a', id: preview.link.id, paper_id: paperId, status: 'accepted', role: 'editor' };
  } }); stores.push(join);
  await join.load(); assert.deepEqual(joinCalls.map(item => item.action), ['preview'], 'Viewing a link never accepts it');
  await join.accept(); assert.equal(join.getSnapshot().pending, true);
  assert.deepEqual(await join.accept(), { paperId, role: 'editor' }, 'A view link preserves existing editor permission');
  assert.deepEqual(joinCalls[1], joinCalls[2], 'Uncertain acceptance retries the original request ID');
  const revoked = createRebuttalJoin({ owner: 'recipient_a', token, request: async () => { throw problem(404, 'link_not_found'); } }); stores.push(revoked);
  await revoked.load(); assert.equal(revoked.getSnapshot().invalid, true); assert.equal(await revoked.accept(), undefined);
  const wrongJoin = createRebuttalJoin({ owner: 'different', token, request: async () => preview }); stores.push(wrongJoin);
  await wrongJoin.load(); assert.equal(wrongJoin.getSnapshot().preview, undefined);
  let active = true; const later = deferred();
  const oldJoin = createRebuttalJoin({ owner: 'recipient_a', token, canAct: () => active, request: async (_path, body) => body.action === 'preview' ? preview : later.promise }); stores.push(oldJoin);
  await oldJoin.load(); const accepting = oldJoin.accept(); active = false;
  later.resolve({ user_id: 'recipient_a', id: preview.link.id, paper_id: paperId, status: 'accepted', role: 'viewer' });
  assert.equal(await accepting, undefined, 'An old account cannot navigate after acceptance');

  // Email-only pending recipients have no account identity and coexist in the owner list.
  const project = { ...newProject(), id: paperId }, email = { id: 'email_a', kind: 'email', user: null, email: 'author@example.org', role: 'editor', created_at: '2030-01-01T00:00:00Z', expires_at: '2030-01-31T00:00:00Z', delivery_status: 'unknown' };
  const detail = { project, revision: 1, role: 'owner', owner, members: [{ user: owner, role: 'owner' }], invitations: [email, { ...email, id: 'email_b', email: 'other@example.org' }] };
  assert.deepEqual(readRebuttalPaper(detail).invitations, detail.invitations);
  assert.throws(() => readRebuttalPaper({ ...detail, invitations: [email, { ...email, id: 'email_b' }] }));
  const memberBodies = [];
  const members = createRebuttalCloud({ owner: owner.id, paperId, request: async (_path, body) => {
    if (!body) return detail; memberBodies.push(body); return { ...detail, revision: body.expected_revision + 1, invitations: body.action === 'cancel_invitation' ? [] : [email] };
  } }); stores.push(members); await members.load();
  assert(await members.changeMember({ email: email.email, role: 'editor' })); assert.equal(memberBodies[0].email, email.email);
  assert(await members.changeMember({ action: 'cancel_invitation', invitation_id: email.id })); assert.equal(memberBodies[1].invitation_id, email.id);
  console.log('Rebuttal sharing: fragment handoff expiry, email schema, exact token retries, per-role links, rotation/revocation, explicit acceptance and account guards passed.');
} finally { stores.forEach(store => store.dispose()); await server.close(); }
