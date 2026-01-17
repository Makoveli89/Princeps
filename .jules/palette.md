## 2024-05-22 - Interactive Cards Pattern
**Learning:** High-value interactive cards (like Workspaces) are implemented as `div`s with `onClick`, lacking keyboard accessibility.
**Action:** When touching list/grid components, always check for `onClick` on non-button elements and upgrade them to `role="button"` with keyboard handlers.

## 2026-01-17 - File Upload Drop Zone Accessibility
**Learning:** File upload drop zones must be fully interactive buttons (click/key) wrapping the input, not just visual containers for a label.
**Action:** Use `role="button"`, `tabIndex={0}`, and `useRef` to trigger hidden file inputs from the container, ensuring keyboard users can upload files.
