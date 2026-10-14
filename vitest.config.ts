import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["**/*.test.ts", "**/*.test.tsx"],
    // "Claude outputs" is the download drop-folder: gitignored and already
    // excluded in tsconfig.json, but Vitest's include globs walk it, so a
    // stale copy of lib/avatar/retarget.test.ts sitting there resolved
    // "./retarget" relative to itself and failed the whole run. Keep this
    // list in step with tsconfig's "exclude".
    exclude: ["node_modules", ".next", ".venv", "Claude outputs"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "."),
    },
  },
});
