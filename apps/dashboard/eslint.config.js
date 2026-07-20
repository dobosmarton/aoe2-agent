import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import prettier from "eslint-config-prettier";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    // Generated or vendored: routeTree.gen.ts is written by the router plugin
    // and holds the only `as any` casts in the repo; components/ui is shadcn
    // CLI output that regenerates as `function` + `interface` on every
    // `shadcn add`, so enforcing our conventions there only creates churn.
    ignores: ["dist/**", "src/routeTree.gen.ts", "src/components/ui/**"],
  },
  js.configs.recommended,
  tseslint.configs.recommendedTypeChecked,
  // `.flat.recommended`, not `["recommended-latest"]` — the latter is still the
  // eslintrc-shaped alias in v7 and flat config rejects its array `plugins`.
  // This set is rules-of-hooks plus the React Compiler diagnostics (purity,
  // immutability, preserve-manual-memoization…), so a component the compiler
  // cannot memoise surfaces here instead of silently opting out.
  reactHooks.configs.flat.recommended,
  reactRefresh.configs.vite,
  prettier,
  {
    languageOptions: {
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/explicit-function-return-type": "error",
      "@typescript-eslint/explicit-module-boundary-types": "error",
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/consistent-type-definitions": ["error", "type"],
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "prefer-const": "error",
      // hooks/use-events.ts rethrows a value it caught as `unknown`; wrapping it
      // in an Error to satisfy the rule would destroy the original stack.
      "@typescript-eslint/only-throw-error": ["error", { allowThrowingUnknown: true }],
    },
  },
  {
    // `ImportMetaEnv` / `ImportMeta` merge into Vite's ambient declarations,
    // which only `interface` can do — the one case the type-over-interface
    // rule genuinely cannot cover.
    files: ["src/vite-env.d.ts"],
    rules: { "@typescript-eslint/consistent-type-definitions": "off" },
  },
  {
    // `throw redirect({ to })` is how TanStack Router expresses a navigation
    // from beforeLoad; the thrown value is a Redirect, not an Error, by design.
    files: ["src/routes/**"],
    rules: {
      "@typescript-eslint/only-throw-error": "off",
      // A route module must export `Route` beside its component, so this rule
      // can never be satisfied here. It is also redundant: the router plugin
      // runs with autoCodeSplitting, which already emits each route component
      // as its own chunk — the separation the rule exists to enforce.
      // Layout components still live in src/layouts/, for architectural
      // reasons rather than to appease a linter.
      "react-refresh/only-export-components": "off",
    },
  },
  {
    // These declare `queryOptions()` factories. That call returns a branded
    // type carrying the query's data type through to useQuery, so a
    // hand-written return annotation either drops the brand or restates a
    // large generic — inference is the intended API.
    files: ["src/lib/queries.ts", "src/hooks/use-events.ts"],
    rules: {
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
    },
  },
  {
    // Application code only: vite.config.ts and this file must default-export,
    // that being the shape their loaders require.
    files: ["src/**"],
    rules: {
      "no-restricted-syntax": [
        "error",
        {
          selector: "ExportDefaultDeclaration",
          message: "Use named exports — they rename cleanly and tree-shake.",
        },
      ],
    },
  },
  {
    files: ["*.config.{js,ts}"],
    extends: [tseslint.configs.disableTypeChecked],
  },
);
