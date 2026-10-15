import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom",
  define: { "import.meta.env.VITE_API_ORIGIN": '"https://api.paperleet.test"' } });
const globals = ["document", "createImageBitmap", "fetch"];
const originals = new Map(globals.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
try {
  const { safeAvatarPath, avatarCrop, prepareProfileAvatar, AVATAR_MAX_FILE_BYTES } = await server.ssrLoadModule("/src/profileAvatar.ts");
  const { readProfileEditPolicy, profileEditLimitMessage } = await server.ssrLoadModule("/src/profileEditPolicy.ts");
  const { Avatar, AffiliationBadge } = await server.ssrLoadModule("/src/social.tsx");
  const path = `/api/avatars/${"a".repeat(32)}/${"b".repeat(64)}.jpg`;
  assert.equal(safeAvatarPath(path), path);
  for (const invalid of [null, "", "https://tracker.example/avatar.jpg", `https://api.paperleet.test${path}`,
    `//tracker.example${path}`, "data:image/jpeg;base64,eA==", `${path}?track=1`, `${path}#x`, `${path}\n`,
    path.replace("/avatars/", "/avatars/../"), path.toUpperCase(), path.replace(".jpg", ".svg")]) {
    assert.equal(safeAvatarPath(invalid), undefined, `Reject unsupported avatar source: ${invalid}`);
  }
  const user = { display_name: "Fixture", avatar_color: "#4f7c73", avatar_url: path };
  const html = renderToString(React.createElement(Avatar, { user }));
  assert(html.includes(`src="https://api.paperleet.test${path}"`), "Avatar images must use the configured API origin");
  assert(html.includes('referrerPolicy="no-referrer"'));
  const unsafe = renderToString(React.createElement(Avatar, { user: { ...user, avatar_url: "https://tracker.example/pixel" } }));
  assert(!unsafe.includes("<img") && !unsafe.includes("tracker.example"));
  assert(unsafe.includes(">F<"), "Unsafe image URLs must fall back to the letter avatar");
  const domain = renderToString(React.createElement(AffiliationBadge, { affiliation: { kind: "email_domain", name: "research.example", domain: "research.example" } }));
  assert(domain.includes("research.example") && !domain.includes("<svg"), "A raw email domain must not claim an institution verification badge");

  assert.deepEqual(avatarCrop(800, 400), { x: 200, y: 0, side: 400 });
  assert.deepEqual(avatarCrop(400, 800), { x: 0, y: 200, side: 400 });
  assert.throws(() => avatarCrop(0, 800));
  const decoded = [];
  const draw = [];
  let closed = 0;
  const bitmap = { width: 800, height: 400, close() { closed++; } };
  const canvas = { width: 0, height: 0,
    getContext() { return { fillStyle: "", fillRect(...args) { draw.push(["fill", ...args]); }, drawImage(...args) { draw.push(["draw", ...args]); } }; },
    toDataURL(type, quality) { assert.equal(type, "image/jpeg"); assert.equal(quality, 0.9); return "data:image/jpeg;base64,eA=="; } };
  Object.assign(globalThis, {
    document: { createElement(type) { assert.equal(type, "canvas"); return canvas; } },
    createImageBitmap: async (file) => { decoded.push(file); return bitmap; },
    fetch: () => { throw Error("Selecting an avatar must not perform a network request"); },
  });
  const file = { type: "image/png", size: AVATAR_MAX_FILE_BYTES };
  assert.equal(await prepareProfileAvatar(file), "data:image/jpeg;base64,eA==");
  assert.equal(canvas.width, 256); assert.equal(canvas.height, 256);
  assert.deepEqual(draw[0], ["fill", 0, 0, 256, 256], "Transparency must flatten on a white background");
  assert.deepEqual(draw[1], ["draw", bitmap, 200, 0, 400, 400, 0, 0, 256, 256]);
  assert.equal(closed, 1, "Decoded image memory must be released");
  for (const bad of [{ type: "image/svg+xml", size: 100 }, { type: "image/jpeg", size: AVATAR_MAX_FILE_BYTES + 1 }, { type: "image/webp", size: 0 }]) {
    await assert.rejects(prepareProfileAvatar(bad));
  }
  assert.equal(decoded.length, 1, "Invalid and oversized files must be rejected before decoding");
  globalThis.createImageBitmap = async () => { throw Error("Broken image"); };
  await assert.rejects(prepareProfileAvatar({ type: "image/jpeg", size: 100 }), (error) => error.code === "decode");

  const policy = { username: { interval_seconds: 2592000, next_change_at: null },
    display_name: { interval_seconds: 604800, next_change_at: "2030-10-21T00:00:00Z" },
    avatar: { daily_limit: 3, remaining: 2, reset_at: "2030-10-15T00:00:00Z" } };
  assert.deepEqual(readProfileEditPolicy(policy), policy);
  assert.equal(readProfileEditPolicy(undefined), undefined, "Old servers must not be treated as enforcing a new policy");
  assert.equal(readProfileEditPolicy({ ...policy, avatar: { ...policy.avatar, remaining: 4 } }), undefined);
  assert.equal(readProfileEditPolicy({ ...policy, username: { interval_seconds: 1, next_change_at: null } }), undefined);
  const limit = { code: "profile_edit_limit", field: "display_name", next_change_at: "2030-10-21T00:00:00Z" };
  assert.match(profileEditLimitMessage(limit, false), /昵称.*当前编辑内容/);
  assert.match(profileEditLimitMessage(limit, true), /Display name.*edits are kept/);
  assert.equal(profileEditLimitMessage({ ...limit, field: "email" }, true), undefined);
  assert.equal(profileEditLimitMessage({ ...limit, next_change_at: "bad" }, true), undefined);
  console.log("PASS avatar safe API URLs, exact center crop, 5 MiB validation, local-only preparation, memory cleanup and profile policy messages");
} finally {
  for (const [name, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
  await server.close();
}
