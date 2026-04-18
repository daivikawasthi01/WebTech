```markdown
# Design System Specification: The Obsidian Pulse

## 1. Overview & Creative North Star
**The Creative North Star: "Precision Noir"**

This design system is not merely a dashboard; it is a high-performance clinical instrument. We are moving away from the "SaaS template" aesthetic toward a "Precision Noir" editorial experience. It balances the stark, authoritative weight of pure black with the aggressive energy of crimson accents. 

To break the "standard" look, we utilize **intentional asymmetry** and **tonal layering**. Layouts should feel like a redacted intelligence report—where white space (or "black space") is used as a structural element to isolate critical data points. We favor high-contrast typography scales and overlapping translucent layers to create a sense of infinite depth within a constrained 2D space.

## 2. Colors & Surface Philosophy

### The Tonal Palette
Our palette is rooted in the absence of light, using a Material 3-derived logic to define hierarchy through luminance rather than lines.

*   **Background:** `#0a0a0a` (The Void)
*   **Primary (Accent):** `primary` (#ffb3ad) / Crimson Base (#e53e3e). Use for high-alert metrics and primary actions.
*   **Surface Tiers:**
    *   `surface_container_lowest`: `#0e0e0e` (Deepest depth)
    *   `surface_container_low`: `#1c1b1b` (Standard sectioning)
    *   `surface_container_high`: `#2a2a2a` (Elevated interaction)
*   **Status:** `Success` (#22c55e), `Warning` (#f59e0b), `Error` (#ffb4ab).

### The "No-Line" Rule
Standard 1px borders are largely prohibited for sectioning. We define boundaries through **Background Color Shifts**. A `surface_container_low` section should sit directly against the `#0a0a0a` background to create a "soft" edge. 

### The "Glass & Gradient" Rule
To add soul to the "Obsidian" theme, use **Glassmorphism** for floating menus or modals. 
*   **Specs:** `surface_variant` at 60% opacity with a `20px` backdrop-blur.
*   **CTAs:** Use a subtle linear gradient for primary buttons, transitioning from `inverse_primary` (#b91c24) to `primary_container` (#ff5450) at a 45-degree angle. This prevents the "flat" look and adds a premium, tactile quality.

## 3. Typography: Editorial Authority
We utilize **Inter** not as a generic sans-serif, but as a monospaced-adjacent grotesque to convey technical precision.

*   **Display Stats (3.5rem / 2.25rem):** High-impact metrics. Use `primary` color for "Best Values."
*   **Section Titles (1.375rem):** `title-lg`. Medium weight. These should have generous top-margin to anchor the content below.
*   **Body (0.8125rem / 13px):** `body-md`. Tight leading for a dense, data-rich feel.
*   **Technical Labels (0.625rem / 10px):** `label-sm`. **Always Uppercase.** 1px letter spacing. Use `secondary` (#888) or `muted` (#444) to de-emphasize meta-data.

## 4. Elevation & Depth

### The Layering Principle
Depth is achieved by stacking. 
1.  **Level 0:** Base Background (`#0a0a0a`).
2.  **Level 1:** Content Areas (`surface_container_low`).
3.  **Level 2:** Interactive Cards (`surface_container_high`).
This creates a natural lift. Avoid drop shadows on nested items; hierarchy is inherent in the tint.

### Ambient Shadows
For floating elements (modals/tooltips), use an **Ambient Shadow**:
*   `box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);` 
*   Add a **Ghost Border**: A `1px` stroke using `outline_variant` (#5b403e) at **15% opacity**. This catches the "light" just enough to define the shape without looking like a stroke.

## 5. Components

### Buttons
*   **Primary:** Crimson gradient, `8px` radius. No border. Text is `on_primary` (pure white or off-white).
*   **Secondary:** Ghost style. `1px` border of `outline_variant` at 20%. Text is `primary`.
*   **States:** On hover, increase the `surface_bright` overlay by 10%.

### Cards & Data Modules
*   **Structure:** `12px` border-radius. Background: `#111111`. 
*   **The Divider Rule:** Forbid the use of horizontal rules (`<hr>`). Use `24px` of vertical whitespace or a subtle shift to `#161616` background for internal headers to separate content.

### Input Fields
*   **Base:** `#161616` background. `8px` radius.
*   **Border:** `1px` solid `#222222` (Active state: `1px` solid `#e53e3e`).
*   **Typography:** User input should be `primary_text` (#e5e5e5), while placeholders are `muted` (#444).

### Data Visualization (The Pulse)
*   **Plot Background:** Pure black (`#0a0a0a`).
*   **Gridlines:** `1px` at `#1a1a1a`. Gridlines should be sparse; only use them for the Y-axis.
*   **Primary Line:** Crimson (#e53e3e). Use a `2px` stroke width with a subtle `4px` outer glow (drop-shadow) of the same color to simulate a "neon" data pulse.

### Interactive Chips
*   **Filter Chips:** `surface_container_highest` background. When active, background becomes `primary_container` with `on_primary_container` text.

## 6. Do’s and Don’ts

### Do:
*   **Do** use asymmetrical margins. For example, give a left-aligned chart a wider right margin to create "breathing room" for the eye.
*   **Do** use the `10px` uppercase labels for all non-data metadata.
*   **Do** prioritize "Tonal Layering." If a card looks flat, try making the background behind it one shade darker rather than adding a shadow.

### Don’t:
*   **Don’t** use pure white text (#ffffff). Always use `on_surface` (#e5e2e1) to reduce eye strain in high-contrast dark mode.
*   **Don’t** use 100% opaque borders for container separation. 
*   **Don’t** use standard blue for links. Every interactive element must exist within the Crimson/Obsidian spectrum.
*   **Don’t** use dividers. If you feel you need a line, use space instead.