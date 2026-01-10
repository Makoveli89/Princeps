# Actionable Recommendations

## 1. Frontend Code Quality: ESLint + Prettier + TypeScript

*   **Feature Name:** Standardized React/Vite Linting & Formatting
*   **Source Repo:** https://github.com/eyvindove/vite-react-typescript-eslint-prettier
*   **Why we need it:** `apps/console` completely lacks linting and formatting. This ensures consistent code style and catches errors early in the React/Vite environment.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Copy `.eslintrc.cjs` and `.prettierrc` to `apps/console/`, and update `package.json` with `lint` scripts.

## 2. CI/CD for Frontend: "Web Build" Job

*   **Feature Name:** Frontend CI Pipeline
*   **Source Repo:** https://github.com/actions/starter-workflows/blob/main/ci/node.js.yml
*   **Why we need it:** `.github/workflows/tests.yml` only tests the Python backend. We need to build and lint the frontend on every PR to prevent UI regressions.
*   **"Raidability" Score (1-10):** 10
*   **Implementation Plan:** Add a `web-build` job to `.github/workflows/tests.yml` that runs `npm ci && npm run lint && npm run build`.

## 3. Essential React Hooks Collection

*   **Feature Name:** `use-local-storage`, `use-debounce`, `use-media-query`
*   **Source Repo:** https://github.com/juliencrn/usehooks-ts
*   **Why we need it:** `apps/console/hooks` is almost empty. Standard hooks prevent re-inventing the wheel for common UI patterns like persisting state or handling responsiveness.
*   **"Raidability" Score (1-10):** 8
*   **Implementation Plan:** Copy individual hook files (e.g., `useLocalStorage.ts`) into `apps/console/hooks/`.

## 4. UI Component Library: Shadcn UI "Primitives"

*   **Feature Name:** Shadcn UI Components (`Button`, `Input`, `Dialog`)
*   **Source Repo:** https://github.com/shadcn-ui/ui (specifically `apps/www/registry/default/ui`)
*   **Why we need it:** The project aims for a "gothic and futuristic" aesthetic but lacks a component library (only `use-toast` exists). "Raiding" these unstyled, accessible primitives is faster than building from scratch.
*   **"Raidability" Score (1-10):** 7 (Requires `cn` utility which we have, and `tailwindcss-animate`).
*   **Implementation Plan:** Copy component files (e.g. `button.tsx`) to `apps/console/components/ui/`.

## 5. Automated Dependency Updates: Dependabot

*   **Feature Name:** Dependabot Configuration
*   **Source Repo:** https://github.com/dependabot/dependabot-core (or standard GitHub docs)
*   **Why we need it:** Dependencies (both pip and npm) will rot. Automating PRs for updates keeps security risks low with zero effort.
*   **"Raidability" Score (1-10):** 10
*   **Implementation Plan:** Create `.github/dependabot.yml` with configurations for `pip` (root) and `npm` (`/apps/console`).

## 6. Contributing Guidelines

*   **Feature Name:** `CONTRIBUTING.md` Template
*   **Source Repo:** https://github.com/nayafia/contributing-template
*   **Why we need it:** The repo lacks instructions for new developers (or agents) on how to setup, test, and submit PRs.
*   **"Raidability" Score (1-10):** 10
*   **Implementation Plan:** Create `CONTRIBUTING.md` in the root using the template, adapted for our Python/React stack.

## 7. Frontend Testing: Vitest Setup

*   **Feature Name:** Vitest Configuration
*   **Source Repo:** https://github.com/vitest-dev/vitest/blob/main/examples/react-testing-lib/vitest.config.ts
*   **Why we need it:** There are zero frontend tests. Vitest is native to Vite and allows for fast unit testing of logic and components.
*   **"Raidability" Score (1-10):** 8
*   **Implementation Plan:** Add `vitest.config.ts` to `apps/console` and install `vitest` + `jsdom`.

## 8. Backend Structured Logging: Structlog Config

*   **Feature Name:** Production-Ready Structlog Configuration
*   **Source Repo:** https://github.com/hynek/structlog (specifically docs/examples)
*   **Why we need it:** `server.py` uses logging, but a robust production config ensures JSON output, context binding (request IDs), and proper formatting for observability tools.
*   **"Raidability" Score (1-10):** 6
*   **Implementation Plan:** Create `framework/logging_config.py` adapting a standard structlog setup and import it in `server.py`.

## 9. Pull Request Template

*   **Feature Name:** PR Description Template
*   **Source Repo:** https://github.com/devspace-cloud/devspace/blob/master/.github/PULL_REQUEST_TEMPLATE.md
*   **Why we need it:** Enforces a standard for PRs (Description, Type of change, Checklist) to ensure quality before review.
*   **"Raidability" Score (1-10):** 10
*   **Implementation Plan:** Create `.github/pull_request_template.md`.

## 10. Strict TypeScript Configuration

*   **Feature Name:** `tsconfig.json` Strict Base
*   **Source Repo:** https://github.com/tsconfig/bases/blob/main/bases/react.json
*   **Why we need it:** `apps/console/tsconfig.json` lacks `"strict": true`. Enabling this prevents an entire class of runtime errors.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Update `apps/console/tsconfig.json` to extend a strict base or manually enable strict flags.
