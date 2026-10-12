# Frontend Test Suite (`frontend/tests/`)

This directory houses automated unit and presentation logic test suites executed via the native Node.js test runner (`node --test`).

---

## 1. Test Modules

* [`helpers.test.mjs`](helpers.test.mjs): Verifies numeric formatting, non-finite score rejection, percentage point scaling, URL protocol security sanitization, and placeholder resolution.
* [`presentation.test.mjs`](presentation.test.mjs): Verifies view mode synchronization, Analyst View advanced actions, Bujji assistant mounting, and dashboard tab navigation.

---

## 2. Execution Command

```bash
# From repository root
npm --prefix frontend test

# From frontend directory
cd frontend
npm test
```
