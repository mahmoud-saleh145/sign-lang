import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Local-only trees that are not project source. Without these, `eslint`
    // walks the Python virtualenv and reports unused-variable warnings out of
    // third-party packages (e.g. .venv/.../sklearn/...), which is noise from
    // code this project does not own. The "Claude outputs" / "claude-*"
    // directories are the download drop-folder and are already excluded in
    // tsconfig.json and vitest.config.ts; keep the three lists in step.
    ".venv/**",
    "node_modules/**",
    "Claude outputs/**",
    "claude-commits/**",
    "claude-patches/**",
    // Generated corpora: local-only, multi-GB, never source.
    "processed-karsl*/**",
    "KArSL/**",
    "data/**",
  ]),
]);

export default eslintConfig;
