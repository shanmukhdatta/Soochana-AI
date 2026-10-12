# GitHub Actions CI Workflows (`.github/workflows/`)

This directory contains the automated CI pipelines for pull request validation and deployment readiness.

---

## 1. Workflow Inventory

* [`ci.yml`](ci.yml): Multi-job continuous integration pipeline:
  - **`backend-tests`**: Sets up Python 3.11, installs dependencies, and runs the 121-case PyTest test suite.
  - **`frontend-tests`**: Sets up Node.js 20, runs unit tests, audits API contract bounds, and builds the production Vite bundle.
  - **`container-check`**: Validates multi-container orchestration syntax (`docker compose config`).
