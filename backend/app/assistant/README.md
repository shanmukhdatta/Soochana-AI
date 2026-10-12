# Soochana AI — Bujji Financial AI Assistant (`app/assistant/`)

`app/assistant/` implements the **context-aware conversational risk intelligence copilot (Bujji)** for Soochana AI. It acts as an embedded companion in the React dashboard, explaining financial risk concepts, model predictions, Index Rebalancing (Module A), and Multi-Asset Stress Testing (Module B) to quantitative analysts, risk engineers, portfolio managers, and technical reviewers.

---

## 1. Architectural Scope & Purpose

* **Scope:** Contextual question-answering and methodology explanations powered by high-throughput LLM inference via the Groq Cloud API (`qwen/qwen3.8-27b`).
* **Dual Audience Adaptation:**
  * **Quick View Mode:** Provides high-level executive summaries, actionable risk takeaways, and intuitive analogies without overwhelming the user with raw mathematical vectors.
  * **Analyst View Mode:** Explains exact mathematical formulations (SLSQP convex optimization bounds, Black-Scholes partial derivatives, 12-month ECL accounting separation, XGBoost impact feature engineering).
* **Non-Goals:** Does **not** provide investment advice, buy/sell recommendations, stock-price predictions, or execute live transactions.

---

## 2. Component Structure

| File | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| [`service.py`](service.py) | Groq API client, prompt assembly, context injection, and guardrails | `BujjiAssistantService`, `chat()` |
| [`schemas.py`](schemas.py) | Pydantic request and response schemas for chat messages | `AssistantChatRequest`, `AssistantChatResponse`, `ChatMessage` |
| [`router.py`](router.py) | FastAPI endpoint mount for the assistant | `/assistant/chat` |

---

## 3. Dynamic Context Injection Architecture

```text
User Question + Active UI Context (Mode: Quick/Analyst, Signal ID: optional)
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│           1. Context Builder (`service.py`)            │
│  - Formats audience mode (Quick vs Analyst instruction)│
│  - Resolves signal metadata from SQLite if signal_id   │
│    is provided (Headline, Sentiment, Event, Impact)    │
│  - Injects system prompt with Soochana AI architecture │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│             2. Groq Cloud API Integration              │
│  - Model: qwen/qwen3.8-27b                             │
│  - Low-latency inference via httpx                     │
│  - Temperature: 0.2 (deterministic risk reasoning)     │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│             3. Structured Response & Audit             │
│  - Clean markdown output returned to React chat widget │
│  - Fallback: Graceful degradation if key is absent     │
│  - Disclaimer: Research and educational use only       │
└────────────────────────────────────────────────────────┘
```

---

## 4. Financial Accuracy & Safety Guardrails

The assistant prompt incorporates strict behavioral rules to prevent financial hallucinations:

1. **Anti-Advisory Guardrail:** Explicit prohibition against issuing personalized buy/sell signals or guaranteed returns.
2. **Attribution Transparency:** When queried regarding XGBoost severity decisions, explains the joint contribution of event probabilities and pre-publication market volatility without inventing per-token attribution.
3. **Prompt Injection Hardening:** Disregards adversarial instructions embedded within external news headlines or user queries.
4. **Credential Isolation:** Never leaks backend environment variables, file paths, or private API keys.

---

## 5. API Contract

### Request: `POST /assistant/chat`
```json
{
  "message": "Why is Apple's rebalancing weight reduced in Module A?",
  "mode": "analyst",
  "signal_id": "692f95dd74396d5d3808",
  "conversation_history": []
}
```

### Response
```json
{
  "response": "In Module A, Apple's (AAPL) target allocation is tilted downward because...",
  "model": "qwen/qwen3.8-27b",
  "mode_used": "analyst",
  "signal_referenced": "692f95dd74396d5d3808"
}
```
