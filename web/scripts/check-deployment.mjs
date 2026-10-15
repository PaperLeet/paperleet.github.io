import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

for (const [publicMode, appAvailable, hostedEdition, hostedUrl] of [
  [false, false, false, ""], [false, true, true, ""],
  [true, true, false, "https://app.paperleet.com"],
  [true, false, false, ""], [true, false, false, "not a URL"],
]) {
  const server = await createServer({
    server: { middlewareMode: true }, appType: "custom",
    optimizeDeps: { noDiscovery: true, entries: [] },
    define: {
      "import.meta.env.VITE_PUBLIC_SITE": JSON.stringify(publicMode ? "1" : "0"),
      "import.meta.env.VITE_HOSTED_APP_AVAILABLE": JSON.stringify(appAvailable ? "1" : "0"),
      "import.meta.env.VITE_HOSTED_APP_URL": JSON.stringify(hostedUrl),
      "import.meta.env.VITE_APP_EDITION": JSON.stringify(hostedEdition ? "hosted" : "local"),
    },
  });
  try {
    const deployment = await server.ssrLoadModule("/src/deployment.ts");
    const { AccountNav } = await server.ssrLoadModule("/src/social.tsx");
    const { signInHref } = await server.ssrLoadModule("/src/accountNavigation.ts");
    const login = signInHref("architecture/attention");
    const expected = publicMode ? appAvailable ? `https://app.paperleet.com/${login}` : "#/" : login;
    assert.equal(deployment.hostedAppHref(login), expected);
    const nav = renderToString(React.createElement(AccountNav, { returnTo: "architecture/attention" }));
    if (appAvailable) assert(nav.includes(expected));
    else assert.equal(nav, "", "Public-only sites must not expose a broken sign-in link");
    for (const hash of ["#/account", login, "#/account%2Fcontinue%2F22", "#/community?sort=recent", "#/comm%75nity", "#/community/problem/architecture/attention", "#/u/researcher", "#/submit"])
      assert(deployment.isHostedRoute(hash));
    for (const hash of ["#/", "#/22", "#/papers/transformer", "#/collections/llm-basics", "#/architecture/attention"])
      assert(!deployment.isHostedRoute(hash));
    for (const url of ["", "not a URL", "http://app.paperleet.com", "https://user:secret@app.paperleet.com", "https://app.paperleet.com/path", "https://app.paperleet.com/?next=evil", "https://app.paperleet.com/#login"])
      assert.throws(() => deployment.validateHostedAppUrl(url));
    if (publicMode || !hostedEdition) {
      const redirects = [], listeners = {};
      globalThis.location = { hash: login, replace: url => redirects.push(url), assign: url => redirects.push(url) };
      globalThis.document = { addEventListener: (type, listener) => { listeners[type] = listener; } };
      globalThis.window = { addEventListener: (type, listener) => { listeners[type] = listener; } };
      if (appAvailable) {
        assert.equal(deployment.startPublicNavigation(), false, "Direct login routes must redirect before rendering forms");
        assert.deepEqual(redirects, [expected]);
        globalThis.location.hash = "#/22";
        assert.equal(deployment.startPublicNavigation(), true);
        globalThis.location.hash = "#/community";
        listeners.hashchange();
        assert.equal(redirects.at(-1), "https://app.paperleet.com/#/community");
      } else {
        assert.equal(deployment.startPublicNavigation(), true);
        assert.equal(globalThis.location.hash, "#/", "Closed account routes must normalize before mounting forms");
        globalThis.location.hash = "#/community";
        listeners.hashchange();
        assert.equal(globalThis.location.hash, "#/");
        assert.deepEqual(redirects, [], "Closed features must not navigate to the unpublished backend");
        if (!publicMode) {
          globalThis.location.hash = "#/submit";
          listeners.hashchange();
          assert.equal(globalThis.location.hash, "#/submit", "The offline paper workflow must remain available locally");
        }
      }
      delete globalThis.location; delete globalThis.document; delete globalThis.window;
    }
  } finally { await server.close(); }
}
console.log("PASS public/hosted login navigation, protected-route redirects before forms, practice routes and fixed HTTPS destination");
