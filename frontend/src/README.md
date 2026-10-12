# Frontend Source Architecture (`frontend/src/`)

`frontend/src/` houses the React 18, TypeScript, and Vite 6 source code for the **Soochana AI Interactive Risk Dashboard**.

---

## 1. Component Hierarchy & Information Flow

```text
                               ┌────────────────────────────────┐
                               │           `main.tsx`           │
                               │        React DOM Root          │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │           `App.tsx`            │
                               │      Root Dashboard Shell      │
                               │ (Global Mode, Tab Router, State│
                               └───────┬──────────────┬─────────┘
                                       │              │
                   ┌───────────────────┴───┐          └───────────────────────┐
                   ▼                       ▼                                  ▼
      ┌─────────────────────────┐ ┌─────────────────────────┐   ┌───────────────────────────┐
      │     Home Landing View   │ │     Live Dashboard      │   │   Bujji AI Copilot Modal  │
      │   (`HomeLanding.tsx`)   │ │      (`App.tsx`)        │   │        (`App.tsx`)        │
      │ • Institutional overview│ │ • 01 / Event Stream     │   │ • Groq Cloud API Client   │
      │ • View mode explainer   │ │ • 02 / Index Lab (Mod A)│   │ • Dynamic context builder │
      │ • Mathematical summary  │ │ • 03 / Stress Lab(Mod B)│   │ • Quick vs Analyst prompt │
      └─────────────────────────┘ └─────────────────────────┘   └───────────────────────────┘
```

---

## 2. File Inventory

* [`main.tsx`](main.tsx): React application entry point and DOM root renderer.
* [`App.tsx`](App.tsx): Core dashboard interface containing navigation headers, tab routing, Quick View and Analyst View layouts, signal tables, Module A rebalancing controls, Module B stress levers, and the Bujji assistant dialog.
* [`api.ts`](api.ts): Typed HTTP client interfacing with the FastAPI backend endpoints (`/analyze`, `/module-a/*`, `/module-b/*`, `/signals`).
* [`types.ts`](types.ts): TypeScript domain definitions (`Signal`, `RebalanceResponse`, `StressResponse`, `Portfolio`).
* [`demo-data.ts`](demo-data.ts): Standalone seed demonstration state and offline client fallbacks.
* [`demo-fixtures.json`](demo-fixtures.json): Pre-computed offline verification payloads for instant preview.
* [`mono.css`](mono.css) & [`styles.css`](styles.css): High-performance monochrome design system with glassmorphic accents, high-contrast badges, and responsive viewports.
* [`components/`](components/README.md): Dedicated reusable view components.
* [`lib/`](lib/README.md): Pure mathematical and formatting utility functions.
