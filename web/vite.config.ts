import { defineConfig, loadEnv, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";
import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { validateDeploymentEnvironment } from "./src/deploymentOrigins";
function publicBrowserAssets(server: Pick<ViteDevServer, "middlewares">) {
  server.middlewares.use((request, response, next) => {
    if (/^\/browser-(runtime|judge)\//.test((request as { url?: string }).url || "")) {
      response.setHeader("Access-Control-Allow-Origin", "*");
      response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    }
    next();
  });
}
function watchBrowserWorker(server: ViteDevServer) {
  const watched = new Set([
    "src/browserWorker.mjs", "src/browserCalibration.ts",
    "public/browser-judge/diagnostics.mjs", "public/browser-judge/packages.mjs", "public/browser-judge/syntax.mjs",
  ].map(path => resolve(server.config.root, path)));
  let generating = Promise.resolve();
  server.watcher.add([...watched]);
  server.watcher.on("change", path => {
    if (!watched.has(resolve(path))) return;
    generating = generating.then(() => new Promise<void>(done => {
      execFile(process.execPath, ["scripts/prepare-browser-worker.mjs"], {cwd:server.config.root}, error => {
        if (error) server.config.logger.error(`Browser worker compilation failed: ${error.message}`);
        else server.ws.send({type:"full-reload"});
        done();
      });
    }));
  });
}
export default defineConfig(({mode}) => {
  const pathRouting = {...loadEnv(mode, process.cwd(), ''), ...process.env}.VITE_PATH_ROUTING === '1';
  return {
  plugins: [react(), {
    name: "browser-judge-public-assets",
    transformIndexHtml(html) {
      return pathRouting ? html.replace('<html lang="en">', '<html lang="en" data-routing="path">') : html;
    },
    configResolved(config) {
      if (config.command === "build") validateDeploymentEnvironment(config.env);
    },
    configureServer(server) { publicBrowserAssets(server); watchBrowserWorker(server); },
    configurePreviewServer: publicBrowserAssets,
  }],
  base: pathRouting ? '/' : './',
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  build: {
    manifest: true,
    rollupOptions: {
      output: {
        onlyExplicitManualChunks: true,
        manualChunks(id) {
          if (id.includes("codemirror") || id.includes("@lezer"))
            return "editor";
          if (
            id.includes("katex") ||
            id.includes("remark") ||
            id.includes("rehype") ||
            id.includes("micromark") ||
            id.includes("mdast") ||
            id.includes("hast")
          )
            return "markdown";
        },
      },
    },
  },
}; });
