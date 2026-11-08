import assert from 'node:assert/strict';
import { createServer } from 'vite';

const server = await createServer({
  configFile: false, server: { middlewareMode: true, watch: null, hmr: false, ws: false }, appType: 'custom',
  optimizeDeps: { noDiscovery: true, include: [] },
  plugins: [{ name: 'rebuttal-cloud-api', enforce: 'pre',
    resolveId(source) { if (source === './social') return '\0rebuttal-cloud-api'; },
    load(id) { if (id === '\0rebuttal-cloud-api') return 'export const socialApi=()=>{throw Error("Unexpected live request")};'; },
  }],
});
const stores = [];
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const problem = (status, code) => Object.assign(Error(code), { status, code });
try {
  const { createRebuttalCloud, createRebuttalInvitations, readRebuttalInvitations, validateCloudProject, readRebuttalRoute, readRebuttalPaper } = await server.ssrLoadModule('/src/rebuttalCloud.ts');
  const { newProject } = await server.ssrLoadModule('/src/rebuttalModel.ts');
  const user = { id: 'account_a', username: 'author_a', display_name: 'Author A' };
  const colleague = { id: 'account_b', username: 'author_b', display_name: 'Author B' };
  const initial = { ...newProject(), id: 'paper_a', title: 'Paper A' };
  const detail = (project = initial, revision = 1, role = 'owner', extra = []) => ({ project: structuredClone(project), revision, role, owner: user, members: [{ user, role: 'owner' }, ...extra] });
  const memory = () => { const values = new Map(); return { values, getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }; };
  const create = options => { const store = createRebuttalCloud({ owner: user.id, paperId: initial.id, debounceMs: 60_000, ...options }); stores.push(store); return store; };

  assert.deepEqual(readRebuttalRoute('rebuttal/paper_a'), { paperId: 'paper_a' });
  assert.deepEqual(readRebuttalRoute('#/rebuttal'), {});
  assert.deepEqual(readRebuttalRoute('#/rebuttal/invitations'), { tab: 'invitations' }, 'Notification landing is never parsed as a paper ID');
  assert.equal(readRebuttalRoute('rebuttal/../paper'), undefined);
  assert.throws(() => readRebuttalPaper(detail(), 'another_paper'));
  assert.throws(() => readRebuttalPaper({ ...detail(), members: [] }));
  assert.deepEqual(readRebuttalPaper(detail()).invitations, [], 'Old servers omit pending invitations');
  const pendingMember = { id: 'invite_a', user: colleague, role: 'editor', created_at: '2030-01-01T00:00:00Z' };
  assert.deepEqual(readRebuttalPaper({ ...detail(), invitations: [pendingMember] }).invitations, [pendingMember]);
  assert.throws(() => readRebuttalPaper({ ...detail(), invitations: [{ ...pendingMember, role: 'owner' }] }));
  assert.throws(() => readRebuttalPaper({ ...detail(initial, 1, 'owner', [{ user: colleague, role: 'editor' }]), invitations: [pendingMember] }));
  const large = { ...initial, opening: '中'.repeat(70_000) };
  assert.throws(() => validateCloudProject(large), /200 KiB/);
  assert.equal(validateCloudProject(large, false).opening, large.opening);

  // Updates made during a request must queue behind it, with the acknowledged revision.
  const first = deferred(), requests = [], savedBodies = [];
  const serial = create({ request: async (_path, body) => {
    if (!body) return detail();
    requests.push(body); savedBodies.push(structuredClone(body));
    if (requests.length === 1) return first.promise;
    return detail({ ...body.project, updatedAt: '2030-01-02T00:00:00.000Z' }, body.expected_revision + 1);
  } });
  await serial.load(); serial.update({ ...initial, title: 'First draft' });
  const flushing = serial.flush();
  serial.update({ ...initial, title: 'Latest draft' });
  assert.equal(requests.length, 1);
  first.resolve(detail({ ...requests[0].project, updatedAt: '2030-01-01T00:00:00.000Z' }, 2));
  assert.equal(await flushing, true);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].expected_revision, 2);
  assert.equal(requests[1].expected_user_id, user.id);
  assert.notEqual(requests[0].request_id, requests[1].request_id);
  assert.equal(serial.getSnapshot().project.title, 'Latest draft');
  assert.equal(serial.getSnapshot().status, 'saved');
  assert.equal(serial.getSnapshot().reloadVersion, 0, 'Autosave must not remount the editor');
  serial.update({ ...serial.getSnapshot().project, updatedAt: initial.updatedAt });
  assert.equal(serial.getSnapshot().dirty, false, 'Server timestamps are not author edits');

  // A timeout after commit is retried with the exact original request, before newer edits.
  const storage = memory(), uncertainBodies = []; let uncertain = true;
  const retry = create({ storage, request: async (_path, body) => {
    if (!body) return detail();
    uncertainBodies.push(structuredClone(body));
    if (uncertain) { uncertain = false; throw Error('network disconnected'); }
    return detail(body.project, body.expected_revision + 1);
  } });
  await retry.load(); retry.update({ ...initial, title: 'May have committed' });
  assert.equal(await retry.flush(), false);
  assert.equal(retry.getSnapshot().status, 'error');
  retry.update({ ...initial, title: 'Still editing offline' });
  assert(storage.values.size > 0);
  assert.equal(await retry.flush(), true);
  assert.deepEqual(uncertainBodies[1], uncertainBodies[0]);
  assert.equal(uncertainBodies[2].project.title, 'Still editing offline');
  assert.equal(storage.values.size, 0, 'A confirmed save clears its recovery copy');

  const conflictStorage = memory(); let remote = detail(); let conflictPosts = 0;
  const conflict = create({ storage: conflictStorage, request: async (_path, body) => {
    if (!body) return remote;
    conflictPosts++; throw problem(409, 'revision_conflict');
  } });
  await conflict.load(); conflict.update({ ...initial, title: 'Preserve this draft' });
  await conflict.flush();
  assert.equal(conflict.getSnapshot().status, 'conflict');
  assert.equal(conflict.getSnapshot().project.title, 'Preserve this draft');
  assert.equal(await conflict.flush(), false); assert.equal(conflictPosts, 1, 'Conflict cannot overwrite with a new revision');
  remote = detail({ ...initial, title: 'Collaborator version' }, 2);
  await conflict.refresh();
  assert.equal(conflict.getSnapshot().project.title, 'Preserve this draft');
  assert.equal(await conflict.refresh(true), true);
  assert.equal(conflict.getSnapshot().project.title, 'Collaborator version');
  assert.equal(conflict.getSnapshot().reloadVersion, 1);
  assert.equal(conflictStorage.values.size, 0);

  const viewerBodies = [];
  const viewer = create({ request: async (_path, body) => { if (body) viewerBodies.push(body); return detail(initial, 1, 'viewer'); } });
  await viewer.load();
  assert.equal(viewer.update({ ...initial, title: 'Forbidden' }), false);
  assert.equal(await viewer.changeMember({ username: colleague.username, role: 'editor' }), false);
  assert.equal(viewerBodies.length, 0);

  const oversizedBodies = [];
  const oversized = create({ request: async (_path, body) => { if (body) oversizedBodies.push(body); return detail(body?.project || initial, body ? body.expected_revision + 1 : 1); } });
  await oversized.load(); assert.equal(oversized.update(large), true);
  assert.equal(oversized.getSnapshot().project.opening, large.opening);
  assert.equal(await oversized.flush(), false); assert.equal(oversizedBodies.length, 0);
  oversized.update({ ...large, opening: 'Short enough' });
  assert.equal(await oversized.flush(), true);

  // Switching accounts invalidates responses and never sends a queued follow-up for the old owner.
  let current = user.id; const switchedRequest = deferred(), switchedBodies = [];
  const switched = create({ canAct: () => current === user.id, request: async (_path, body) => {
    if (!body) return detail(); switchedBodies.push(body); return switchedRequest.promise;
  } });
  await switched.load(); switched.update({ ...initial, title: 'Old account draft' });
  const switchFlush = switched.flush(); switched.update({ ...initial, title: 'Queued old draft' });
  current = colleague.id; switchedRequest.resolve(detail(switchedBodies[0].project, 2));
  assert.equal(await switchFlush, false); assert.equal(switchedBodies.length, 1);
  assert.equal(switched.getSnapshot().document.revision, 1);

  // React StrictMode's setup-cleanup-setup must restart reading and ignore the first response.
  const staleRead = deferred(); let strictReads = 0;
  const strict = create({ request: async () => ++strictReads === 1 ? staleRead.promise : detail({ ...initial, title: 'Current mount' }, 2) });
  const initialLoad = strict.load(); strict.dispose(); strict.activate(); await strict.load();
  staleRead.resolve(detail({ ...initial, title: 'Unmounted response' })); await initialLoad;
  assert.equal(strict.getSnapshot().project.title, 'Current mount');
  assert.equal(strict.getSnapshot().loading, false);

  // A stale poll cannot create a false conflict after our own save has advanced the revision.
  const pollRead = deferred(); let pollReads = 0;
  const polling = create({ request: async (_path, body) => body ? detail(body.project, body.expected_revision + 1) : ++pollReads === 1 ? detail() : pollRead.promise });
  await polling.load(); const poll = polling.refresh();
  polling.update({ ...initial, title: 'Our revision' }); await polling.flush();
  pollRead.resolve(detail()); await poll;
  assert.equal(polling.getSnapshot().status, 'saved'); assert.equal(polling.getSnapshot().document.revision, 2);

  // Recovery is account scoped and preserves the frozen idempotent request across remounts.
  const recoveredStorage = memory(), recoveredBodies = [];
  const lost = create({ storage: recoveredStorage, request: async (_path, body) => { if (!body) return detail(); recoveredBodies.push(body); throw Error('offline'); } });
  await lost.load(); lost.update({ ...initial, title: 'Recovered text' }); await lost.flush(); lost.dispose();
  const resumed = create({ storage: recoveredStorage, request: async (_path, body) => { if (!body) return detail(); recoveredBodies.push(body); return detail(body.project, body.expected_revision + 1); } });
  await resumed.load(); assert.equal(resumed.getSnapshot().project.title, 'Recovered text');
  await resumed.flush(); assert.deepEqual(recoveredBodies[1], recoveredBodies[0]);
  const anotherAccount = create({ owner: colleague.id, storage: recoveredStorage, request: async () => detail() });
  await anotherAccount.load(); assert.equal(anotherAccount.getSnapshot().project.title, 'Paper A');

  // Membership mutations serialize after saves, allow exact retries, and do not freeze typo failures.
  const memberBodies = []; let memberAttempts = 0;
  const members = create({ request: async (path, body) => {
    if (!body) return detail(); memberBodies.push({ path, body: structuredClone(body) });
    if (path.endsWith('/members') && ++memberAttempts === 1) throw Error('lost membership response');
    return detail(body.project || initial, body.expected_revision + 1, 'owner', [{ user: colleague, role: 'editor' }]);
  } });
  await members.load(); members.update({ ...initial, title: 'Before invite' });
  assert.equal(await members.changeMember({ username: colleague.username, role: 'editor' }), false);
  assert.equal(memberBodies[0].path.endsWith('/members'), false);
  assert.equal(memberBodies[1].body.expected_revision, 2);
  assert.equal(members.hasPendingMember(), true);
  assert.equal(await members.changeMember(), true);
  assert.deepEqual(memberBodies[2], memberBodies[1]);
  assert.equal(members.getSnapshot().document.members.length, 2);

  const typo = create({ request: async (_path, body) => { if (!body) return detail(); throw problem(404, 'member_not_found'); } });
  await typo.load(); assert.equal(await typo.changeMember({ username: 'missing_user', role: 'viewer' }), false);
  assert.equal(typo.hasPendingMember(), false); assert.equal(typo.getSnapshot().blocked, false);
  assert.equal(typo.getSnapshot().status, 'error');

  const denied = create({ request: async (_path, body) => { if (!body) return detail(); throw problem(409, 'account_changed'); } });
  await denied.load(); denied.update({ ...initial, title: 'Keep on account mismatch' }); await denied.flush();
  assert.equal(denied.getSnapshot().blocked, true); assert.equal(denied.getSnapshot().project.title, 'Keep on account mismatch');
  const inviteA = { id: 'invite_a', paper_id: 'paper_a', title: 'Paper A', venue: 'ICLR', owner: colleague, role: 'editor', created_at: '2030-01-01T00:00:00Z' };
  const inviteB = { ...inviteA, id: 'invite_b', paper_id: 'paper_b', title: 'Paper B', role: 'viewer' };
  const invitationList = invitations => ({ user_id: user.id, invitations });
  const invitationStore = options => { const store = createRebuttalInvitations({ owner: user.id, ...options }); stores.push(store); return store; };
  assert.deepEqual(readRebuttalInvitations(invitationList([inviteA]), user.id), [inviteA]);
  assert.throws(() => readRebuttalInvitations({ ...invitationList([inviteA]), user_id: colleague.id }, user.id), /account changed/);
  assert.throws(() => readRebuttalInvitations(invitationList([{ ...inviteA, role: 'owner' }]), user.id));

  // No paper access is fabricated: accepting/declining only removes the confirmed invitation.
  const responseBodies = [];
  const decisions = invitationStore({ request: async (path, body) => {
    if (!body) return invitationList([inviteA, inviteB]);
    responseBodies.push({ path, body });
    return { id: path.endsWith('invite_a') ? inviteA.id : inviteB.id, paper_id: path.endsWith('invite_a') ? inviteA.paper_id : inviteB.paper_id,
      status: body.action === 'accept' ? 'accepted' : 'declined' };
  } });
  await decisions.load();
  assert.deepEqual(await decisions.respond(inviteA, 'accept'), { id: inviteA.id, paper_id: inviteA.paper_id, status: 'accepted' });
  assert.deepEqual(decisions.getSnapshot().invitations, [inviteB]);
  assert.equal(responseBodies[0].body.expected_user_id, user.id);
  assert.equal((await decisions.respond(inviteB, 'decline')).status, 'declined');
  assert.deepEqual(decisions.getSnapshot().invitations, []);

  // A lost response locks competing actions and retries the exact same accept request.
  const lostDecision = deferred(), decisionBodies = [];
  const decisionRetry = invitationStore({ request: async (_path, body) => {
    if (!body) return invitationList([inviteA, inviteB]);
    decisionBodies.push(structuredClone(body));
    if (decisionBodies.length === 1) return lostDecision.promise;
    return { id: inviteA.id, paper_id: inviteA.paper_id, status: 'accepted' };
  } });
  await decisionRetry.load(); const accepting = decisionRetry.respond(inviteA, 'accept');
  assert.equal(await decisionRetry.respond(inviteB, 'decline'), undefined);
  lostDecision.reject(Error('response lost')); await accepting;
  assert.equal(decisionRetry.getSnapshot().pending, true);
  assert.equal(decisionRetry.getSnapshot().invitations.length, 2);
  assert.equal(await decisionRetry.respond(inviteA, 'decline'), undefined);
  assert.equal((await decisionRetry.respond()).status, 'accepted');
  assert.deepEqual(decisionBodies[0], decisionBodies[1]);
  assert.equal(decisionRetry.getSnapshot().pending, false);

  // An older GET cannot resurrect an invitation after its action succeeds.
  const oldInvitationRead = deferred(); let invitationReads = 0;
  const invitationPolling = invitationStore({ request: async (_path, body) => body ? { id: inviteA.id, paper_id: inviteA.paper_id, status: 'declined' } :
    ++invitationReads === 1 ? invitationList([inviteA]) : oldInvitationRead.promise });
  await invitationPolling.load(); const pendingPoll = invitationPolling.load();
  await invitationPolling.respond(inviteA, 'decline');
  oldInvitationRead.resolve(invitationList([inviteA])); await pendingPoll;
  assert.deepEqual(invitationPolling.getSnapshot().invitations, []);

  // A response from the previous account cannot navigate or mutate its successor's state.
  let invitationOwner = user.id; const switchedInvitation = deferred();
  const invitationSwitch = invitationStore({ canAct: () => invitationOwner === user.id, request: async (_path, body) => body ? switchedInvitation.promise : invitationList([inviteA]) });
  await invitationSwitch.load(); const oldAction = invitationSwitch.respond(inviteA, 'accept');
  invitationOwner = colleague.id; switchedInvitation.resolve({ id: inviteA.id, paper_id: inviteA.paper_id, status: 'accepted' });
  assert.equal(await oldAction, undefined); assert.equal(invitationSwitch.getSnapshot().invitations.length, 1);
  const wrongAccount = invitationStore({ request: async () => ({ user_id: colleague.id, invitations: [inviteA] }) });
  await wrongAccount.load(); assert.deepEqual(wrongAccount.getSnapshot().invitations, []); assert.match(wrongAccount.getSnapshot().error, /account changed/);

  const cancelled = invitationStore({ request: async (_path, body) => { if (!body) return invitationList([inviteA]); throw problem(404, 'invitation_not_found'); } });
  await cancelled.load(); await cancelled.respond(inviteA, 'accept');
  assert.equal(cancelled.getSnapshot().pending, false); assert.deepEqual(cancelled.getSnapshot().invitations, []);
  let unsupportedReads = 0;
  const oldServer = invitationStore({ request: async () => { unsupportedReads++; throw problem(404, 'not_found'); } });
  await oldServer.load(); await oldServer.load();
  assert.equal(oldServer.getSnapshot().unavailable, true); assert.equal(unsupportedReads, 1, 'Do not repeatedly poll an old API');
  console.log('Rebuttal cloud: serial saves, retries, conflicts, recovery, account isolation, roles, UTF-8 limits, invitations and lifecycle passed.');
} finally {
  stores.forEach(store => store.dispose());
  await server.close();
}
await import('./check-rebuttal-invitations.mjs');
await import('./check-rebuttal-sharing.mjs');
