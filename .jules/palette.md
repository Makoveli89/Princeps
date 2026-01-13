## 2026-01-13 - Icon-only Buttons in Chat Interface
**Learning:** The chat interface uses icon-only buttons (Send, Netrunner Toggle) which completely lacked accessibility labels, making the primary interaction point invisible to screen readers.
**Action:** Always audit `lucide-react` icon usage in `button` elements to ensure they are accompanied by `aria-label` or `title` attributes, especially in high-traffic areas like the chat input.
