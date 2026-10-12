# Frontend Automation & Verification Scripts (`frontend/scripts/`)

This directory houses Node.js verification scripts, API contract checkers, and end-to-end integration audits.

---

## 1. Scripts Inventory

* [`audit-e2e.mjs`](audit-e2e.mjs): Comprehensive 8-phase end-to-end integration audit executed against live running servers:
  1. API Health & Readiness checks
  2. Live signal feed verification
  3. Module A baseline index retrieval
  4. Module A tactical rebalancing calculation
  5. Module B synthetic portfolio retrieval
  6. Module B adverse stress calculation
  7. Module B What-If scaling simulation
  8. Module B Reverse Stress testing
* [`check-contracts.mjs`](check-contracts.mjs): Verifies that all 15 FastAPI backend route contracts are preserved without client-side drift.
* [`capture-audit-screenshots.mjs`](capture-audit-screenshots.mjs): Automated headless browser screenshot runner.
* [`test-disconnect.mjs`](test-disconnect.mjs): Offline resilience verification testing client behavior when the backend server is unreachable.
