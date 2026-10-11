# Soochana AI — Bujji Financial AI Assistant

`app/assistant/` implements the **context-aware conversational risk assistant (Bujji)** for Soochana AI. It acts as an embedded companion in the React dashboard, explaining complex financial risk concepts, model predictions, Index Rebalancing (Module A), and Multi-Asset Stress Testing (Module B) to analysts and hackathon judges.

---

## 1. Architectural Scope & Purpose

* **Scope:** Contextual question-answering and methodology explanations powered by high-throughput LLM inference via the Groq Cloud API.
* **Dual Audience Adaptation:**
  * **Quick View Mode:** Provides high-level executive summaries, actionable takeaways, and intuitive analogies without overwhelming the user with raw mathematical vectors.
  * **Analyst View Mode:** Explains exact formulas (SLSQP convex optimization bounds, Black-Scholes partial derivatives, 12-month ECL accounting separation, XGBoost impact feature engineering).
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
User Question + Active UI Context (Mode, Signal ID)
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│           1. Context Builder (service.py)              │
│  - Formats audience mode (Quick vs Analyst)            │
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
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│             3. Structured Response & Audit             │
│  - Clean markdown output returned to React chat widget │
│  - Disclaimer: Research and educational use only       │
└────────────────────────────────────────────────────────┘
```

---

## 4. Financial Accuracy & Safety Guardrails

The assistant prompt incorporates strict behavioral rules to prevent financial hallucinations:

1. **Strict No-Advice Policy:**
   Bujji explicitly identifies as an educational intelligence assistant and will never offer personalized financial advice or forecast asset price returns.
2. **Honest Attribution Limits:**
   If asked why a specific impact score was assigned to an event, Bujji explains the composite XGBoost mapping (FinBERT probabilities + sentiment + market volatility). If token-level feature attribution was not persisted for that signal, it explicitly states that per-token SHAP/LIME attribution is not stored rather than fabricating explanations.
3. **Graceful Degradation:**
   If `GROQ_API_KEY` is omitted or invalid in `.env`, the endpoint returns a clear, transparent message (`Groq API key not configured; assistant unavailable`) rather than throwing an unhandled exception or fabricating canned responses.

---

## 5. API Route Contracts

* `POST /assistant/chat`:
  * **Request Body:**
    ```json
    {
      "messages": [{"role": "user", "content": "Explain why AAPL has a 7.09% post-rebalance weight."}],
      "mode": "analyst",
      "signal_id": "optional-signal-uuid"
    }
    ```
  * **Response Body:**
    ```json
    {
      "reply": "In Module A, AAPL's weight increased from 6.67% to 7.09%...",
      "model": "qwen/qwen3.8-27b",
      "status": "success"
    }
    ```
