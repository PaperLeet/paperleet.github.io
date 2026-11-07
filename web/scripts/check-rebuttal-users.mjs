import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { USERS_KEY, readUsers, writeUsers, createLocalUser, selectLocalUser, renameLocalUser, workspaceStorage } =
    await server.ssrLoadModule("/src/rebuttalUsers.ts");
  const { WORKSPACE_KEY, newProject, readWorkspace, writeWorkspace } = await server.ssrLoadModule("/src/rebuttalModel.ts");
  const memory = new Map();
  const storage = { getItem: key => memory.get(key) ?? null, setItem: (key, value) => { memory.set(key, value); } };
  const legacyProject = { ...newProject(), title: "Existing paper" };
  const legacyWorkspace = { version: 1, activeProjectId: legacyProject.id, projects: [legacyProject] };
  assert(writeWorkspace(storage, legacyWorkspace));
  const legacyBytes = memory.get(WORKSPACE_KEY);

  const initial = readUsers(storage);
  assert.equal(initial.error, false);
  assert.equal(initial.directory.activeUserId, "local");
  assert.equal(initial.directory.users.length, 1);
  assert.equal(memory.has(USERS_KEY), false, "Reading must not persist a profile directory implicitly");
  assert.deepEqual(readWorkspace(workspaceStorage(storage, "local")).workspace, legacyWorkspace);
  assert.equal(memory.get(WORKSPACE_KEY), legacyBytes, "Existing workspace data stays in its original key");
  assert.equal(memory.has("paperleet.rebuttal.user.local.v1"), false, "Do not duplicate legacy data");

  const originalDirectory = structuredClone(initial.directory);
  let directory = createLocalUser(initial.directory, "  Researcher B  ");
  const secondId = directory.activeUserId;
  assert.notEqual(secondId, "local");
  assert.equal(directory.users[1].name, "Researcher B");
  assert.deepEqual(initial.directory, originalDirectory, "Profile updates must not mutate React state");
  const secondStorage = workspaceStorage(storage, secondId);
  const secondWorkspace = readWorkspace(secondStorage).workspace;
  assert.notEqual(secondWorkspace.activeProjectId, legacyWorkspace.activeProjectId);
  secondWorkspace.projects[0].title = "Separate paper";
  assert(writeWorkspace(secondStorage, secondWorkspace));
  assert(memory.has(`paperleet.rebuttal.user.${secondId}.v1`));
  assert.deepEqual(readWorkspace(secondStorage).workspace, secondWorkspace);
  assert.equal(memory.get(WORKSPACE_KEY), legacyBytes, "Saving another profile must not change legacy data");

  directory = selectLocalUser(directory, "local");
  assert.equal(directory.activeUserId, "local");
  directory = renameLocalUser(directory, secondId, "  Renamed researcher  ");
  assert.equal(directory.users[1].name, "Renamed researcher");
  assert.equal(directory.users[1].id, secondId);
  assert.deepEqual(readWorkspace(workspaceStorage(storage, secondId)).workspace, secondWorkspace);
  assert(writeUsers(storage, directory));
  assert.deepEqual(readUsers(storage), { directory, error: false });

  for (const name of ["", "   ", "x".repeat(81), null]) {
    assert.throws(() => createLocalUser(directory, name));
    assert.throws(() => renameLocalUser(directory, "local", name));
  }
  assert.throws(() => selectLocalUser(directory, "missing"));
  assert.throws(() => renameLocalUser(directory, "missing", "Name"));
  const boundary = createLocalUser(directory, "x".repeat(80));
  assert.equal(boundary.users.at(-1).name.length, 80);
  let full = directory;
  while (full.users.length < 20) full = createLocalUser(full, `User ${full.users.length + 1}`);
  assert.equal(new Set(full.users.map(user => user.id)).size, 20);
  assert.throws(() => createLocalUser(full, "Overflow"));

  assert.throws(() => secondStorage.getItem(USERS_KEY));
  assert.throws(() => secondStorage.setItem(`paperleet.rebuttal.user.local.v1`, "bad"));
  assert.throws(() => workspaceStorage(storage, "../local"));
  assert.throws(() => workspaceStorage(storage, ""));

  const directoryBytes = memory.get(USERS_KEY);
  assert(!writeUsers({ ...storage, setItem() { throw Error("quota"); } }, directory));
  assert.equal(memory.get(USERS_KEY), directoryBytes);
  assert(readUsers({ getItem() { throw Error("denied"); }, setItem() {} }).error);
  assert.throws(() => workspaceStorage({ getItem() { throw Error("denied"); }, setItem() {} }, secondId).getItem(WORKSPACE_KEY));

  for (const mutate of [
    value => { value.version = 2; },
    value => { value.activeUserId = "missing"; },
    value => { value.users = []; },
    value => { value.users.push(value.users[0]); },
    value => { value.users[0].id = "invalid.id"; },
    value => { value.users[0].name = "  "; },
    value => { value.users[0].createdAt = "yesterday"; },
    value => { value.users = [...full.users, { ...full.users[0], id: "overflow" }]; },
  ]) {
    const invalid = structuredClone(directory);
    mutate(invalid);
    const raw = JSON.stringify(invalid);
    memory.set(USERS_KEY, raw);
    assert(readUsers(storage).error);
    assert.equal(memory.get(USERS_KEY), raw, "Invalid directory reads preserve recoverable bytes");
    assert(!writeUsers(storage, invalid));
    assert.equal(memory.get(USERS_KEY), raw, "Invalid directory writes leave existing storage intact");
  }
  for (const raw of ["{broken", "null", "[]", "x".repeat(64_001)]) {
    memory.set(USERS_KEY, raw);
    const result = readUsers(storage);
    assert(result.error);
    assert.equal(result.directory.activeUserId, "local");
    assert.equal(memory.get(USERS_KEY), raw);
  }
  assert.equal(memory.get(WORKSPACE_KEY), legacyBytes);
  assert.deepEqual(readWorkspace(secondStorage).workspace, secondWorkspace);
  console.log("PASS local user profiles, legacy workspace preservation, isolated paper storage and directory recovery");
} finally {
  await server.close();
}
