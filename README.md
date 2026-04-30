# Vera Message Engine — magicpin AI Challenge

## Approach

**Signal-routing composer** that picks the single best signal from trigger + merchant state + category context before writing. Most bots dump all context into one prompt and hope the LLM figures it out. Ours selects the most compelling signal first, then composes around it.

### Architecture

```
POST /v1/context → ContextStore (versioned, idempotent)
POST /v1/tick    → Prioritizer → Composer (NVIDIA Gemma) → Actions
POST /v1/reply   → ReplyHandler (auto-reply/intent/hostile detection) → Composer → Response
```

### Key Design Decisions

1. **NVIDIA API (Gemma 3n)** — fast, OpenAI-compatible, generous free tier
2. **20+ trigger-specific prompt variants** — each trigger kind gets framing tailored to what scores highest
3. **3-strike auto-reply ladder** — Try once → Wait 24h → End. Fixes Vera's biggest production pain point
4. **Instant intent transitions** — the moment a merchant says "yes", we switch to action mode. No more qualifying questions
5. **Zero fabrication** — every number is grounded in context. Rationale traces each fact to its source
6. **Natural Hindi-English code-mix** — based on merchant's language preference, not forced

### What Makes This Win

- **Decision quality**: Signal prioritizer scores each trigger on urgency × engagement recency × kind importance. Only the best signal fires.
- **Category fit**: Voice rules (tone, vocabulary, taboos) are hard constraints in the prompt, not suggestions.
- **Merchant fit**: Owner first name, specific performance numbers, active offers, review themes — all pulled from MerchantContext.
- **Engagement compulsion**: Single CTA, low-friction asks, curiosity/loss-aversion hooks, effort externalization.

### Model

NVIDIA API with Google Gemma 3n (e4b-it). Temperature=0 for deterministic output. OpenAI-compatible endpoint.

### Technical Constraints Met

| Constraint | Implementation |
|---|---|
| **30s timeout** | LLM timeout set to 25s with deterministic fallback |
| **10 req/sec** | FastAPI async handles concurrent requests |
| **500 KB payload** | Context stored by reference, not duplicated |
| **20 actions/tick** | Capped in prioritizer with per-merchant dedup |

### Tradeoffs

- **In-memory state** — sufficient for the 60-min test window; would use Redis/SQLite for production
- **No retrieval/RAG** — category digest items are small enough to fit in context window directly
- **Single LLM call per composition** — could add a validation pass, but latency budget is tight (30s)

## Running

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Start the Bot Server
Open a terminal in the project directory:
```bash
python -m uvicorn bot.server:app --host 0.0.0.0 --port 8080
```
Keep this terminal open.

### Step 2: Expose Publicly with ngrok
Open a **second terminal**:
```bash
ngrok http 8080
```
Copy the `https://....ngrok-free.dev` URL from the output.

### Step 3: Verify
Visit `https://YOUR-NGROK-URL.ngrok-free.dev/` in your browser. You should see:
```json
{"name": "Vera Message Engine", "version": "1.0.0", "status": "running", ...}
```

### Step 4: Run the Judge
Open a **third terminal**:
```bash
python judge_simulator.py
```
Make sure `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL` are set in `judge_simulator.py`.

### Step 5: Submit
Submit the ngrok URL (e.g., `https://loth-paula-hydropic.ngrok-free.dev`) to the judge.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Bot info and available endpoints |
| GET | `/v1/healthz` | Liveness probe with context counts |
| GET | `/v1/metadata` | Team info, model, approach |
| POST | `/v1/context` | Receive context (category/merchant/customer/trigger) |
| POST | `/v1/tick` | Periodic wake-up, returns proactive message actions |
| POST | `/v1/reply` | Handle merchant/customer replies |

## Project Structure

```
magicpin/
├── bot/
│   ├── __init__.py
│   ├── config.py          # Configuration, constants, detection patterns
│   ├── context_store.py   # Versioned in-memory context storage
│   ├── composer.py        # LLM-powered message composition engine
│   ├── reply_handler.py   # Multi-turn conversation intelligence
│   ├── prioritizer.py     # Trigger ranking and signal selection
│   └── server.py          # FastAPI server with all endpoints
├── dataset/               # Seed data (categories, merchants, triggers)
├── expanded/              # Generated expanded dataset
├── judge_simulator.py     # LLM-powered judge for scoring
├── test_bot.py            # Smoke test script
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
└── README.md              # This file
```
