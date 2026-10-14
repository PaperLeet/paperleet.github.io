import assert from "node:assert/strict";
import { createServer } from "vite";

const originalFetch = globalThis.fetch;
for (const origin of ["", "https://app.paperleet.com"]) {
  const server = await createServer({
    server: { middlewareMode: true }, appType: "custom",
    optimizeDeps: { noDiscovery: true, entries: [] },
    define: { "import.meta.env.VITE_API_ORIGIN": JSON.stringify(origin) },
  });
  try {
    const { apiFetch, apiUrl, validateApiOrigin } = await server.ssrLoadModule("/src/apiClient.ts");
    const { socialApi } = await server.ssrLoadModule("/src/social.tsx");
    const { requestJob } = await server.ssrLoadModule("/src/judgeJobs.ts");
    const requests = [];
    globalThis.fetch = async (url, options) => {
      requests.push({ url, options });
      return Response.json({ user: { id: "owner" } });
    };
    assert.equal((await socialApi("account/me")).user.id, "owner");
    await socialApi("account/profile", { display_name: "Researcher" });
    await apiFetch("/api/health");
    await apiFetch("/api/submissions", { method: "POST", body: "{}" });
    await apiFetch("/api/practice/stats", { credentials: "omit" });
    await assert.rejects(requestJob("", "client-1", new AbortController().signal, { problem: "gae" }, async (url, options) => {
      requests.push({ url, options });
      return Response.json({ error: "Sign in" }, { status: 401 });
    }), /Sign in/);
    assert.deepEqual(requests.map(r => r.url), ["account/me", "account/profile", "health", "submissions", "practice/stats", "judge/jobs"].map(path => origin + "/api/" + path));
    requests.forEach((r, i) => assert.equal(r.options.credentials, i === 4 ? "omit" : origin ? "include" : "same-origin"));
    assert.equal(requests[1].options.method, "POST");
    assert.deepEqual(JSON.parse(requests[1].options.body), { display_name: "Researcher" });
    assert.equal(requests[5].options.headers["X-PaperLeet-Client-ID"], "client-1");
    for (const value of ["https://evil.test/api/account/me", "//evil.test/api/account/me", "/api\\evil", "/other"])
      assert.throws(() => apiUrl(value));
    for (const value of ["http://app.paperleet.com", "https://user:secret@app.paperleet.com", "https://app.paperleet.com/api", "https://app.paperleet.com/?next=evil", "https://app.paperleet.com/#/"])
      assert.throws(() => validateApiOrigin(value));
  } finally {
    globalThis.fetch = originalFetch;
    await server.close();
  }
}
console.log("PASS same-origin and split-origin account, submission, health, statistics and judge requests with API-host cookies");
