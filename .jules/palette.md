## 2024-05-22 - Interactive Cards Pattern
**Learning:** High-value interactive cards (like Workspaces) are implemented as `div`s with `onClick`, lacking keyboard accessibility.
**Action:** When touching list/grid components, always check for `onClick` on non-button elements and upgrade them to `role="button"` with keyboard handlers.

## 2026-01-16 - Workspace Form Accessibility
**Learning:** Forms often lack accessible labels and feedback mechanisms (loading states, disabled states).
**Action:** Always wrap inputs in labels or use htmlFor. Provide clear loading indicators for async actions to prevent double-submissions.
