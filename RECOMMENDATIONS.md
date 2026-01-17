# Actionable Recommendations

## 1. Feature Name: Frontend Testing Infrastructure (Vitest)

*   **Source Repo:** https://github.com/vitest-dev/vitest
*   **Why we need it:** The `apps/console` frontend has zero unit tests. Vitest is the Vite-native standard for fast, reliable component and logic testing, closing a major quality gap.
*   **"Raidability" Score (1-10):** 8
*   **Implementation Plan:** Add `vitest.config.ts` to `apps/console` and install `vitest`, `@testing-library/react`, and `jsdom`.

## 2. Feature Name: Tailwind Class Merger (`cn` utility)

*   **Source Repo:** https://github.com/shadcn-ui/ui/blob/main/apps/www/lib/utils.ts
*   **Why we need it:** The codebase is missing `lib/utils.ts` (causing build errors) and lacks a robust way to handle conditional Tailwind classes, which is essential for dynamic UI components.
*   **"Raidability" Score (1-10):** 10
*   **Implementation Plan:** Copy the 6-line `cn` function (using `clsx` and `tailwind-merge`) into `apps/console/lib/utils.ts`.

## 3. Feature Name: Local Development Orchestration (Docker Compose)

*   **Source Repo:** https://github.com/tiangolo/full-stack-fastapi-template
*   **Why we need it:** Currently, developers must run backend and frontend separately. A `docker-compose.yml` unifies the stack (FastAPI + Postgres/Pgvector + React) for a reproducible "one-command" start.
*   **"Raidability" Score (1-10):** 7
*   **Implementation Plan:** Create `docker-compose.yml` at the root defining `backend`, `frontend`, and `db` services.

## 4. Feature Name: Backend Testing Configuration (Pytest)

*   **Source Repo:** https://github.com/intility/fastapi-azure-auth/blob/main/pytest.ini
*   **Why we need it:** The backend lacks a standardized `pytest.ini`, leading to inconsistent test execution and warning noise. A solid config ensures strict markers and correct python paths.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Add `pytest.ini` to the root with `[pytest]` configuration for `asyncio_mode` and `pythonpath`.

## 5. Feature Name: Frontend Pre-commit Hooks

*   **Source Repo:** https://github.com/pre-commit/pre-commit-hooks
*   **Why we need it:** The current `.pre-commit-config.yaml` ignores the `apps/console` directory. We need to enforce linting (`eslint`) and formatting (`prettier`) on frontend files before they enter the repo.
*   **"Raidability" Score (1-10):** 8
*   **Implementation Plan:** Add a `system` hook entry to `.pre-commit-config.yaml` that runs `cd apps/console && npm run lint` on staged JS/TS files.

## 6. Feature Name: Type-Safe API Client Generator

*   **Source Repo:** https://github.com/ferdikoomen/openapi-typescript-codegen
*   **Why we need it:** The frontend manually `fetch`es data with loose typing. Generating a client from FastAPI's `openapi.json` guarantees strict type safety and auto-completion for all API endpoints.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Add a script in `apps/console/package.json` to run `openapi --input http://localhost:8000/openapi.json --output ./src/client`.

## 7. Feature Name: Request Correlation ID Middleware

*   **Source Repo:** https://github.com/snok/asgi-correlation-id
*   **Why we need it:** The backend logs are hard to trace per-request. This middleware attaches a unique `X-Request-ID` to every log entry and response header, simplifying debugging in distributed systems.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Install `asgi-correlation-id` and add `app.add_middleware(CorrelationIdMiddleware)` in `server.py`.

## 8. Feature Name: Type-Safe Configuration (Pydantic Settings)

*   **Source Repo:** https://github.com/pydantic/pydantic-settings
*   **Why we need it:** `server.py` relies on raw `os.getenv` calls, which are error-prone and untyped. Pydantic Settings provides validation, type hints, and centralized management for environment variables.
*   **"Raidability" Score (1-10):** 6
*   **Implementation Plan:** Create `brain/core/settings.py` defining a `Settings` class and replace `os.getenv` usage in `server.py`.

## 9. Feature Name: React Error Boundary Component

*   **Source Repo:** https://github.com/bvaughn/react-error-boundary
*   **Why we need it:** If a React component crashes, the entire app goes white. An Error Boundary catches these errors and displays a user-friendly fallback UI instead of a blank screen.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Wrap the `<App />` component in `apps/console/index.tsx` with `<ErrorBoundary FallbackComponent={ErrorFallback}>`.

## 10. Feature Name: Lint Staged (Optimized Git Hooks)

*   **Source Repo:** https://github.com/okonet/lint-staged
*   **Why we need it:** Running linters on the entire codebase during pre-commit is slow. `lint-staged` ensures we only check the files that actually changed, significantly speeding up the commit workflow.
*   **"Raidability" Score (1-10):** 9
*   **Implementation Plan:** Install `lint-staged` in `apps/console`, add configuration to `package.json`, and invoke it from `.pre-commit-config.yaml`.
