# Raid Report: Princeps Brain V2

**Agent:** Github Repo Raider V2
**Target:** React (Vite) + FastAPI Stack
**Focus:** "Delight" Features, Stability, and Modern Tooling

Below are 10 high-value, portable assets found in the public ecosystem that fill critical gaps in your current repository.

---

### 1. Feature: The `cn` Utility (Shadcn/UI Core)
**Source Repo:** [shadcn-ui/ui](https://github.com/shadcn-ui/ui/blob/main/apps/www/lib/utils.ts)
**Why we need it:** Your React frontend uses Tailwind CSS (`apps/console`). As components grow, conditional class logic becomes messy (`${active ? 'bg-red' : ''}`). The `cn` utility (ClassNames + TailwindMerge) is the industry standard for clean, conflict-free style composition.
**"Raidability" Score:** 10/10 (Single function, copy-paste).
**"Value" Score:** 9/10 (Foundational for all UI work).
**Implementation Plan:** Create `apps/console/lib/utils.ts` and paste the function.

### 2. Feature: `use-toast` Hook (Feedback System)
**Source Repo:** [shadcn-ui/ui](https://github.com/shadcn-ui/ui/blob/main/apps/www/registry/default/ui/use-toast.ts)
**Why we need it:** The "Princeps Console" lacks a unified way to show feedback (e.g., "Agent Started", "Ingestion Failed"). This hook provides a global, accessible toast notification system that looks "gothic and futuristic" out of the box.
**"Raidability" Score:** 8/10 (Requires copying the hook and the `Toaster` component).
**"Value" Score:** 9/10 (Critical for UX).
**Implementation Plan:** Copy to `apps/console/hooks/use-toast.ts` and `apps/console/components/ui/toaster.tsx`.

### 3. Feature: Structlog Middleware (Structured Logging)
**Source Repo:** [hynek/structlog](https://github.com/hynek/structlog) (and standard FastAPI recipes)
**Why we need it:** `server.py` currently uses `print()` statements. This is unreadable in production. Structlog formats logs as JSON (machine-readable) with context (request IDs, latency), making debugging async AI chains possible.
**"Raidability" Score:** 9/10 (Install lib, copy config block).
**"Value" Score:** 10/10 (Essential for AI observability).
**Implementation Plan:** Add `structlog` to `requirements.txt` and add middleware block to `server.py`.

### 4. Feature: Global Exception Handler
**Source Repo:** [tiangolo/fastapi](https://fastapi.tiangolo.com/tutorial/handling-errors/)
**Why we need it:** Your `server.py` relies on repetitive `try/except` blocks inside every endpoint. A global handler catches *everything* (even unpredicted bugs), logs the stack trace securely, and returns a clean "Internal Error" JSON to the client.
**"Raidability" Score:** 10/10 (Copy `@app.exception_handler` block).
**"Value" Score:** 8/10 (Code cleanup + Security).
**Implementation Plan:** Add the handler function to `server.py` and remove inline try/excepts.

### 5. Feature: `Justfile` (Modern Task Runner)
**Source Repo:** [casey/just](https://github.com/casey/just)
**Why we need it:** You have `start.sh` and `start.bat`. A `Justfile` replaces both with a single, cross-platform syntax. It's the modern, developer-friendly alternative to `Makefile` for mixed Python/JS repos.
**"Raidability" Score:** 10/10 (Create one file).
**"Value" Score:** 7/10 (DevEx improvement).
**Implementation Plan:** Create `Justfile` in the root with `run`, `install`, and `test` recipes.

### 6. Feature: `SWR` (Stale-While-Revalidate)
**Source Repo:** [vercel/swr](https://github.com/vercel/swr)
**Why we need it:** Your React app needs to poll for Agent status updates. `SWR` handles polling, caching, and revalidation automatically with one hook (`useSWR`). It eliminates the need for complex `useEffect` data fetching logic.
**"Raidability" Score:** 10/10 (Install lib, one-line import).
**"Value" Score:** 9/10 (Massively simplifies frontend state).
**Implementation Plan:** Run `npm i swr` in `apps/console` and replace fetch calls.

### 7. Feature: `SlowAPI` (Rate Limiting)
**Source Repo:** [laurentS/slowapi](https://github.com/laurentS/slowapi)
**Why we need it:** AI endpoints are expensive. If the console is exposed (even internally), a bug could loop and drain your credits. `SlowAPI` adds a decorator `@limiter.limit("5/minute")` to protect critical routes.
**"Raidability" Score:** 9/10 (Install lib, add middleware).
**"Value" Score:** 8/10 (Cost protection).
**Implementation Plan:** Add `slowapi` to `requirements.txt` and configure in `server.py`.

### 8. Feature: GitHub Issue Templates
**Source Repo:** [stevemao/github-issue-templates](https://github.com/stevemao/github-issue-templates)
**Why we need it:** As the project grows, "It's broken" bug reports waste time. Templates force reporters to provide "Steps to Reproduce" and "Environment", streamlining your "Deep Planning" mode for future tasks.
**"Raidability" Score:** 10/10 (Copy markdown files).
**"Value" Score:** 6/10 (Process improvement).
**Implementation Plan:** Create `.github/ISSUE_TEMPLATE/bug_report.md`.

### 9. Feature: `ThemeProvider` (Dark Mode Support)
**Source Repo:** [pacocoursey/next-themes](https://github.com/pacocoursey/next-themes) (works with Vite)
**Why we need it:** You requested a "gothic and futuristic" aesthetic. This requires a robust dark mode implementation that persists user preference and avoids "flash of unstyled content".
**"Raidability" Score:** 9/10 (Install lib, wrap App component).
**"Value" Score:** 7/10 (Aesthetic/UX).
**Implementation Plan:** Install `next-themes` in `apps/console` and wrap `App.tsx`.

### 10. Feature: Dependabot Config
**Source Repo:** [dependabot/dependabot-core](https://github.com/dependabot/dependabot-core)
**Why we need it:** The AI ecosystem moves fast (LangChain, OpenAI SDKs update weekly). Dependabot automatically opens PRs to keep your dependencies secure and up-to-date without manual checking.
**"Raidability" Score:** 10/10 (Copy one yaml file).
**"Value" Score:** 8/10 (Security/Maintenance).
**Implementation Plan:** Create `.github/dependabot.yml`.
