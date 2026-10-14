import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

for (const publicMode of [false, true]) {
  const server = await createServer({
    server: { middlewareMode: true }, appType: "custom",
    optimizeDeps: { noDiscovery: true, entries: [] },
    define: {
      "import.meta.env.VITE_PUBLIC_SITE": JSON.stringify(publicMode ? "1" : "0"),
      "import.meta.env.VITE_HOSTED_APP_URL": JSON.stringify("https://app.paperleet.com"),
    },
  });
  try {
    const deployment = await server.ssrLoadModule("/src/deployment.ts");
    const { AccountNav } = await server.ssrLoadModule("/src/social.tsx");
    const { signInHref } = await server.ssrLoadModule("/src/accountNavigation.ts");
    const login = signInHref("architecture/attention");
    const expected = publicMode ? `https://app.paperleet.com/${login}` : login;
    assert.equal(deployment.hostedAppHref(login), expected);
    assert(renderToString(React.createElement(AccountNav, { returnTo: "architecture/attention" })).includes(expected));
    for (const hash of ["#/account", login, "#/community/problem/architecture/attention", "#/u/researcher", "#/submit"])
      assert(deployment.isHostedRoute(hash));
    for (const hash of ["#/", "#/22", "#/papers/transformer", "#/collections/llm-basics", "#/architecture/attention"])
      assert(!deployment.isHostedRoute(hash));
    for (const url of ["http://app.paperleet.com", "https://user:secret@app.paperleet.com", "https://app.paperleet.com/path", "https://app.paperleet.com/?next=evil", "https://app.paperleet.com/#login"])
      assert.throws(() => deployment.validateHostedAppUrl(url));
    if (publicMode) {
      const redirects = [], listeners = {};
      globalThis.location = { hash: login, replace: url => redirects.push(url), assign: url => redirects.push(url) };
      globalThis.document = { addEventListener: (type, listener) => { listeners[type] = listener; } };
      globalThis.window = { addEventListener: (type, listener) => { listeners[type] = listener; } };
      assert.equal(deployment.startPublicNavigation(), false, "Direct login routes must redirect before rendering forms");
      assert.deepEqual(redirects, [expected]);
      globalThis.location.hash = "#/22";
      assert.equal(deployment.startPublicNavigation(), true);
      globalThis.location.hash = "#/community";
      listeners.hashchange();
      assert.equal(redirects.at(-1), "https://app.paperleet.com/#/community");
      delete globalThis.location; delete globalThis.document; delete globalThis.window;
    }
  } finally { await server.close(); }
}
console.log("PASS public/hosted login navigation, protected-route redirects before forms, practice routes and fixed HTTPS destination");
