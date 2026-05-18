# Frontend Changes

## Feature: Dark/Light Theme Toggle

### Files Modified

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

---

### `frontend/index.html`

Added a `<button id="themeToggle">` element just before the closing `</body>` tag. The button contains two inline SVG icons:

- **Moon icon** (`.icon-moon`) — visible in dark mode; clicking switches to light mode.
- **Sun icon** (`.icon-sun`) — visible in light mode; clicking switches to dark mode.

The button has `aria-label="Toggle light/dark theme"` and `title` for accessibility and keyboard navigation.

---

### `frontend/style.css`

1. **Dark theme variables (`:root`)** — Expanded with new semantic variables for consistent theming:
   - `--welcome-shadow` — shadow for the welcome message bubble
   - `--code-bg` — inline code and code block backgrounds (`rgba(0,0,0,0.25)`)
   - `--error-color / --error-bg / --error-border` — error message palette
   - `--success-color / --success-bg / --success-border` — success message palette

2. **Light theme variables (`[data-theme="light"]`)** — Full accessible light-mode palette:
   - Background: `#f8fafc`; Surface: `#ffffff`; Surface hover: `#f1f5f9`
   - Text primary: `#0f172a` (WCAG AA on white); Text secondary: `#475569`
   - Primary blue shifted slightly darker (`#1d4ed8`) to stay AA-contrast on light backgrounds
   - Border: `#cbd5e1` (visible but not harsh)
   - Code blocks: `rgba(15,23,42,0.07)` (soft blue-gray tint on white)
   - Error: dark red `#b91c1c`; Success: dark green `#15803d` — both WCAG AA on white
   - Welcome message: blue-tinted surface `#eff6ff` with `#93c5fd` border
   - Toggle button: `#e2e8f0` / `#cbd5e1` hover

3. **Hardcoded color fixes** — Replaced three hardcoded values that bypassed the variable system:
   - `rgba(0,0,0,0.2)` on `.message-content code` → `var(--code-bg)`
   - `rgba(0,0,0,0.2)` on `.message-content pre` → `var(--code-bg)`
   - `0 4px 16px rgba(0,0,0,0.2)` on welcome message → `var(--welcome-shadow)`
   - Hardcoded `#f87171` / `#4ade80` on error/success → `var(--error-color)` / `var(--success-color)` etc.

4. **Bug fix** — `blockquote` referenced the nonexistent `var(--primary)`; corrected to `var(--primary-color)`.

5. **Toggle button styles** (`.theme-toggle`) — Fixed-position circular button in the top-right corner (`top: 1rem; right: 1rem; z-index: 100`). Includes:
   - Hover: scale up + darker background
   - Focus: `box-shadow` focus ring matching the app's existing focus style
   - Active: scale down for tactile feel
   - `transition` on background, border, transform, and box-shadow

6. **Icon cross-fade animation** — Replaced the `display: none/block` icon swap with an `opacity + transform` approach so the icons animate smoothly. Both SVGs are stacked via `position: absolute; top: 50%; left: 50%` and centered with `translate(-50%, -50%)`. In dark mode the moon fades in at `rotate(0deg)` and the sun fades out at `rotate(90deg)`. In light mode the roles reverse with `rotate(-90deg)` on exit. Transition: `opacity 0.25s ease, transform 0.25s ease`.

7. **Button click feedback** — Added `@keyframes toggleBounce` (scale to 0.82 then back to 1) and `.theme-toggle--spinning` class applied by JS on each click, auto-removed on `animationend`.

8. **Global theme-transition class** — Added `.theme-transition` selector covering `*, *::before, *::after` that forces `background-color`, `color`, `border-color`, and `box-shadow` transitions (`!important`) on every element. Applied for exactly 300 ms by JS during each theme switch, then removed — so normal interaction transitions are unaffected outside that window.

9. **Smooth body/sidebar transitions** — `body` transitions `background-color` and `color`; `.sidebar` transitions `background-color` and `border-color` (these persist permanently as a baseline even without the `.theme-transition` window).

---

### `frontend/script.js`

Added three functions called before `setupEventListeners()` on `DOMContentLoaded`:

- **`initTheme()`** — Reads `localStorage.getItem('theme')` on page load. Falls back to the OS preference via `window.matchMedia('(prefers-color-scheme: dark)')`. Applies the resolved theme immediately (no transition class, so the initial paint is instant).
- **`applyTheme(theme)`** — Adds `.theme-transition` to `<html>`, sets or removes `data-theme="light"` on `document.documentElement`, writes the choice to `localStorage`, then removes `.theme-transition` after 300 ms via `setTimeout`.
- **`toggleTheme()`** — Adds `.theme-toggle--spinning` to the button (removed on `animationend` via a one-time listener), reads the current `data-theme` attribute, then calls `applyTheme` with the opposite value.

Wired `toggleTheme` to the button's `click` event inside `setupEventListeners()`.
