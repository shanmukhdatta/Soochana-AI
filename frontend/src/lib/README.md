# Frontend Utilities & Helpers (`frontend/src/lib/`)

This directory provides pure JavaScript/TypeScript utility modules for numeric formatting, date parsing, and security sanitization.

---

## 1. Modules Inventory

* [`helpers.mjs`](helpers.mjs): Core mathematical and formatting utilities:
  - `formatNumber()`: Decimal precision formatting
  - `formatPercent()`: Basis point and percentage change conversions
  - `sanitizeUrl()`: Strict URL protocol validation (blocking `javascript:` attacks)
  - `resolveIssuer()`: Safe scenario placeholder resolution
* [`helpers.d.mts`](helpers.d.mts): Complete TypeScript type definitions for the helper modules.
