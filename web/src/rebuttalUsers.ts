import { makeId, WORKSPACE_KEY } from "./rebuttalModel";

export type StorageLike = Pick<Storage, "getItem" | "setItem">;
export type LocalUser = { id: string; name: string; createdAt: string };
export type UserDirectory = { version: 1; activeUserId: string; users: LocalUser[] };
export const USERS_KEY = "paperleet.rebuttal.users.v1";

const MAX_USERS = 20;
const MAX_DIRECTORY_LENGTH = 64_000;

function userId(value: unknown): string {
  if (typeof value !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(value)) throw Error("Invalid local user ID");
  return value;
}

function userName(value: unknown): string {
  if (typeof value !== "string") throw Error("A local user name is required");
  const name = value.trim();
  if (!name || name.length > 80) throw Error("Local user names must be 1–80 characters");
  return name;
}

function validateDirectory(value: unknown): UserDirectory {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("Invalid local user directory");
  const data = value as Record<string, unknown>;
  if (data.version !== 1 || !Array.isArray(data.users) || !data.users.length || data.users.length > MAX_USERS) {
    throw Error("Invalid local user directory");
  }
  const ids = new Set<string>();
  const users = data.users.map((raw: unknown): LocalUser => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw Error("Invalid local user");
    const user = raw as Record<string, unknown>;
    const id = userId(user.id);
    if (ids.has(id)) throw Error("Duplicate local user ID");
    ids.add(id);
    if (typeof user.createdAt !== "string" || user.createdAt.length > 80 || !Number.isFinite(Date.parse(user.createdAt))) {
      throw Error("Invalid local user creation date");
    }
    return { id, name: userName(user.name), createdAt: user.createdAt };
  });
  const activeUserId = userId(data.activeUserId);
  if (!ids.has(activeUserId)) throw Error("The active local user does not exist");
  return { version: 1, activeUserId, users };
}

function defaultDirectory(): UserDirectory {
  return {
    version: 1,
    activeUserId: "local",
    users: [{ id: "local", name: "本地用户", createdAt: new Date().toISOString() }],
  };
}

/** Local profiles organize browser storage; they are not authenticated accounts. */
export function readUsers(storage: StorageLike): { directory: UserDirectory; error: boolean } {
  try {
    const raw = storage.getItem(USERS_KEY);
    if (raw === null) return { directory: defaultDirectory(), error: false };
    if (raw.length > MAX_DIRECTORY_LENGTH) throw Error("Local user directory is too large");
    return { directory: validateDirectory(JSON.parse(raw)), error: false };
  } catch {
    // Leave damaged or unavailable storage untouched so a caller can recover it.
    return { directory: defaultDirectory(), error: true };
  }
}

export function writeUsers(storage: StorageLike, directory: UserDirectory): boolean {
  try {
    const serialized = JSON.stringify(validateDirectory(directory));
    if (serialized.length > MAX_DIRECTORY_LENGTH) return false;
    storage.setItem(USERS_KEY, serialized);
    return true;
  } catch {
    return false;
  }
}

export function createLocalUser(directory: UserDirectory, name: string): UserDirectory {
  const current = validateDirectory(directory);
  const normalizedName = userName(name);
  if (current.users.length >= MAX_USERS) throw Error("At most 20 local users are supported");
  const existing = new Set(current.users.map(user => user.id));
  let id = userId(makeId());
  for (let attempt = 0; existing.has(id); attempt++) {
    if (attempt >= 100) throw Error("Unable to create a unique local user ID");
    id = userId(makeId());
  }
  return {
    version: 1,
    activeUserId: id,
    users: [...current.users, { id, name: normalizedName, createdAt: new Date().toISOString() }],
  };
}

export function selectLocalUser(directory: UserDirectory, id: string): UserDirectory {
  const current = validateDirectory(directory);
  if (!current.users.some(user => user.id === id)) throw Error("The local user does not exist");
  return { ...current, activeUserId: id };
}

export function renameLocalUser(directory: UserDirectory, id: string, name: string): UserDirectory {
  const current = validateDirectory(directory);
  const normalizedName = userName(name);
  if (!current.users.some(user => user.id === id)) throw Error("The local user does not exist");
  return { ...current, users: current.users.map(user => user.id === id ? { ...user, name: normalizedName } : user) };
}

/** Keep the original default workspace in place; route new profiles to distinct keys. */
export function workspaceStorage(storage: StorageLike, id: string): StorageLike {
  const validId = userId(id);
  const scopedKey = validId === "local" ? WORKSPACE_KEY : `paperleet.rebuttal.user.${validId}.v1`;
  const keyFor = (key: string): string => {
    if (key !== WORKSPACE_KEY) throw Error("Unexpected workspace storage key");
    return scopedKey;
  };
  return {
    getItem: key => storage.getItem(keyFor(key)),
    setItem: (key, value) => storage.setItem(keyFor(key), value),
  };
}
