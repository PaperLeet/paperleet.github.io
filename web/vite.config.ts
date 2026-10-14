import { defineConfig, type ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";
function publicBrowserAssets(server: Pick<ViteDevServer, "middlewares">) {
  server.middlewares.use((request, response, next) => {
    if (/^\/browser-(runtime|judge)\//.test((request as { url?: string }).url || "")) {
      response.setHeader("Access-Control-Allow-Origin", "*");
      response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    }
    next();
  });
}
export default defineConfig({
  plugins: [react(), {
    name: "browser-judge-public-assets",
    configureServer: publicBrowserAssets,
    configurePreviewServer: publicBrowserAssets,
  }],
  base: "./",
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  build: {
    rollupOptions: {
      output: {
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
});
