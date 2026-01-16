## 2024-05-24 - Accessible Chat Interfaces
**Learning:** Chat interfaces often rely heavily on icon-only buttons (like a paper plane for "Send" or a sparkler for "AI Mode") which creates a significant barrier for screen reader users if `aria-label` is missing.
**Action:** Always add descriptive `aria-label` and `title` attributes to icon-only buttons, even if the icon itself feels intuitive to sighted users. For toggle buttons, explicit `aria-pressed` state is critical.
