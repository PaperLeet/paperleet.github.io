import assert from "node:assert/strict";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer, resolveConfig } from "vite";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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
    const { AccountNav,AccountProvider,useAccount } = await server.ssrLoadModule("/src/social.tsx");
    const { initialDeploymentHealth,unavailableDeploymentHealth } = await server.ssrLoadModule('/src/deploymentHealth.ts');
    const {practiceOwner} = await server.ssrLoadModule('/src/PracticeWorkspace.tsx');
    const initial = initialDeploymentHealth(hostedEdition);
    assert.equal(practiceOwner(initial?.mode,undefined,true),hostedEdition ? 'unresolved' : 'local','Standalone browser progress must not wait for an optional /api/health service');
    let providerHealth;
    function HealthProbe() {providerHealth = useAccount().health;return null;}
    renderToString(React.createElement(AccountProvider,null,React.createElement(HealthProbe)));
    assert.deepEqual(providerHealth,initial,'The actual provider must use the offline-safe initial health state');
    const connected = {mode:'hosted',auth_required:true,judge:{ready:true}};
    const offline = unavailableDeploymentHealth(connected,hostedEdition);
    assert.equal(offline.mode,hostedEdition ? 'hosted' : 'local');
    assert.equal(offline.judge.ready,false);
    assert.equal(unavailableDeploymentHealth(undefined,true),undefined,'A failed hosted health request must not unlock a local workspace or expose cached grades');
    assert.equal(practiceOwner(unavailableDeploymentHealth(undefined,false).mode),'local');
    assert.equal(connected.judge.ready,true,'Connection recovery cannot mutate previous snapshots');
    const { signInHref } = await server.ssrLoadModule("/src/accountNavigation.ts");
    const login = signInHref("architecture/attention");
    const expected = publicMode ? appAvailable ? `https://app.paperleet.com/${login}` : "#/" : login;
    assert.equal(deployment.hostedAppHref(login), expected);
    const nav = renderToString(React.createElement(AccountNav, { returnTo: "architecture/attention" }));
    assert(nav.includes('account-menu-trigger'),'Preferences remain available from the avatar in every edition');
    const {AccountMenuContent}=await server.ssrLoadModule('/src/HeaderAccountMenu.tsx');
    const menu=renderToString(React.createElement(AccountMenuContent,{returnTo:'architecture/attention',user:null,loading:false,onClose(){}}));
    if (appAvailable) assert(menu.includes(expected));
    else assert(!menu.includes('account-menu-signin'), "Public-only sites must not expose a broken sign-in link");
    assert(menu.includes('href="#/settings"') && menu.includes('href="#/achievements"'),'Browser-only editions retain local settings and achievements');
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

// Resolve the actual Vite build configuration, including mode files and shell
// precedence. This must fail before any browser chunks can be emitted.
const fixture = mkdtempSync(join(tmpdir(), "paperleet-deployment-env-"));
const inherited = Object.fromEntries(Object.entries(process.env).filter(([key]) => key.startsWith("VITE_")));
const modeFile = join(fixture, ".env.production");
function settings(values, file = modeFile) {
  writeFileSync(file, Object.entries(values).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join("\n") + "\n");
}
const resolved = (mode = "production") => resolveConfig({ envDir: fixture, mode, logLevel: "silent" }, "build");
try {
  for (const key of Object.keys(inherited)) delete process.env[key];
  for (const values of [
    {}, { VITE_PUBLIC_SITE: "1" },
    { VITE_PUBLIC_SITE: "1", VITE_HOSTED_APP_AVAILABLE: "0", VITE_HOSTED_APP_URL: "unused-invalid-value" },
    { VITE_PUBLIC_SITE: "1", VITE_HOSTED_APP_AVAILABLE: "1", VITE_HOSTED_APP_URL: "https://app.paperleet.com" },
    { VITE_APP_EDITION: "hosted", VITE_API_ORIGIN: "" },
    { VITE_APP_EDITION: "hosted", VITE_API_ORIGIN: "https://app.paperleet.com" },
  ]) {
    settings(values);
    await resolved();
  }
  const invalid = ["", "not a URL", "http://app.paperleet.com", "https://user:fixture-secret@app.paperleet.com",
    "https://app.paperleet.com/api", "https://app.paperleet.com/?fixture-secret", "https://app.paperleet.com/#fixture-secret"];
  async function refuses(key, mode = "production") {
    await assert.rejects(resolved(mode), error => {
      assert.match(error.message, new RegExp(`^Invalid ${key}:`));
      assert(!error.message.includes("fixture-secret"), "Configuration diagnostics must not echo URL credentials or query data");
      return true;
    });
  }
  for (const value of invalid) {
    settings({ VITE_PUBLIC_SITE: "1", VITE_HOSTED_APP_AVAILABLE: "1", VITE_HOSTED_APP_URL: value });
    await refuses("VITE_HOSTED_APP_URL");
    if (value) {
      settings({ VITE_APP_EDITION: "hosted", VITE_API_ORIGIN: value });
      await refuses("VITE_API_ORIGIN");
    }
  }
  settings({ VITE_PUBLIC_SITE: "1", VITE_HOSTED_APP_AVAILABLE: "1", VITE_HOSTED_APP_URL: "http://wrong.test" }, join(fixture, ".env"));
  settings({ VITE_HOSTED_APP_URL: "https://app.paperleet.com" });
  assert.equal((await resolved()).env.VITE_HOSTED_APP_URL, "https://app.paperleet.com", "Mode files override the base file");
  settings({ VITE_HOSTED_APP_URL: "http://wrong.test" }, join(fixture, ".env.production.local"));
  await refuses("VITE_HOSTED_APP_URL");
  process.env.VITE_HOSTED_APP_URL = "https://app.paperleet.com";
  assert.equal((await resolved()).env.VITE_HOSTED_APP_URL, "https://app.paperleet.com", "Shell values override mode-local files");
  process.env.VITE_HOSTED_APP_URL = "https://user:fixture-secret@app.paperleet.com";
  await refuses("VITE_HOSTED_APP_URL");
  delete process.env.VITE_HOSTED_APP_URL;
  settings({ VITE_PUBLIC_SITE: "1", VITE_HOSTED_APP_AVAILABLE: "1", VITE_HOSTED_APP_URL: "https://app.paperleet.com" }, join(fixture, ".env"));
  settings({ VITE_HOSTED_APP_URL: "http://wrong.test" }, join(fixture, ".env.staging"));
  assert.equal((await resolved("development")).env.VITE_HOSTED_APP_URL, "https://app.paperleet.com");
  await refuses("VITE_HOSTED_APP_URL", "staging");
} finally {
  for (const key of Object.keys(process.env)) if (key.startsWith("VITE_")) delete process.env[key];
  Object.assign(process.env, inherited);
  rmSync(fixture, { recursive: true, force: true });
}
console.log("PASS public/hosted login navigation, protected-route redirects before forms, practice routes and fixed HTTPS destination");
console.log("PASS real build configuration: enabled origins, disabled integration, mode/file/shell precedence and bounded diagnostics");
