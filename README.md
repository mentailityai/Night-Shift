# Night-Shift Preference Optimizer

**An automated data pipeline that captures user corrections to AI outputs and transforms them into reusable training data — so your AI learns from every edit.**

Night-Shift operates on a **dual-loop architecture**:

- **Fast Loop** — Extracted preferences are immediately available to any AI agent via an MCP server. No retraining required.
- **Slow Loop** — Once enough high-quality training pairs accumulate, the system exports JSONL for permanent model fine-tuning.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration Reference](#configuration-reference)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Integrating into an Agentic Workflow](#integrating-into-an-agentic-workflow)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [License](#license)

---

## How It Works

1. A user edits an AI-generated output (e.g., corrects a legal clause, adjusts formatting, fixes a tone issue).
2. Your application captures the original AI output and the human correction and sends them to the Night-Shift API.
3. A background worker (the "Night-Shift Agent") analyses the delta between the two versions using an LLM to extract the underlying preference rule.
4. The extracted rule is embedded and stored in a vector database for **immediate semantic retrieval** by any AI agent (Fast Loop).
5. A cleaned prompt-response training pair is staged for **future model fine-tuning** (Slow Loop).

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                         YOUR APPLICATION                          │
  │                                                                   │
  │   User edits AI output  ──►  POST /api/logs  ──►  Night-Shift    │
  └────────────────────────────────────────────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                    Cron Trigger                 Batch Trigger
                         └─────────────┬─────────────┘
                                       ▼
                           ┌───────────────────────┐
                           │  Night-Shift Agent     │
                           │  (LLM Analysis)        │
                           │                        │
                           │  • Analyse the delta   │
                           │  • Extract the rule    │
                           │  • Clean training data │
                           └───────────┬───────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │   FAST LOOP         │    │   SLOW LOOP         │
              │                     │    │                     │
              │   Extracted Rules   │    │   Training Pairs    │
              │   ────────────────  │    │   ────────────────  │
              │   Stored as vector  │    │   Staged in DB      │
              │   embeddings.       │    │   until threshold   │
              │                     │    │   is met.           │
              │   Queried via MCP   │    │   Exported as JSONL │
              │   by any AI agent   │    │   for fine-tuning   │
              │   before generation.│    │   your model.       │
              └─────────────────────┘    └─────────────────────┘
```

---

## Architecture

| Component | Technology | Purpose |
|---|---|---|
| **API Server** | FastAPI + Uvicorn | Captures interaction payloads from your application |
| **Database** | PostgreSQL + pgvector | Stores logs, rules (with embeddings), and training pairs |
| **Task Queue** | Celery + Redis | Schedules and runs background processing (cron + batch trigger) |
| **LLM Client** | vLLM / OpenAI / Anthropic | Analyses the human correction delta and extracts rules |
| **Embedding Model** | BAAI/bge-m3 (1024-dim) | Generates vector embeddings for semantic rule search |
| **MCP Server** | Model Context Protocol | Exposes preference rules as a tool for AI agents |
| **Fine-Tune Exporter** | JSONL + replay buffer | Exports training data when the configurable threshold is met |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required for type hints and async features |
| Docker & Docker Compose | Latest | For PostgreSQL + pgvector and Redis |
| A running LLM endpoint | — | Local vLLM instance, OpenAI API key, or Anthropic API key |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/NightShift.git
cd NightShift
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the environment

```bash
cp .env.example .env
```

Open `.env` and update the following at minimum:

| Variable | What to set |
|---|---|
| `LLM_PROVIDER` | `vllm`, `openai`, or `anthropic` |
| `VLLM_BASE_URL` | Your vLLM server URL (e.g., `http://192.168.1.180:8000/v1`) |
| `OPENAI_API_KEY` | Your OpenAI key (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Your Anthropic key (if using Anthropic) |

All other settings have sensible defaults. See the full [Configuration Reference](#configuration-reference) below.

### 5. Start infrastructure services

```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) with the pgvector extension pre-installed
- **Redis** (port 6379) as the Celery message broker

### 6. Run database migrations

```bash
alembic upgrade head
```

---

## Configuration Reference

All settings live in the `.env` file. Every parameter has a sensible default so the system works out of the box for local development.

### LLM Provider

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `vllm` | Active provider: `vllm`, `openai`, or `anthropic` |
| `VLLM_BASE_URL` | `http://192.168.1.180:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL_NAME` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Model served by vLLM |
| `VLLM_API_KEY` | `token-placeholder` | vLLM API key (if authentication is enabled) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL_NAME` | `gpt-4o` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL_NAME` | `claude-sonnet-4-20250514` | Anthropic model to use |

### Worker Scheduling

| Variable | Default | Description |
|---|---|---|
| `WORKER_CRON_HOUR` | `2` | Hour (UTC) for the nightly processing run |
| `WORKER_CRON_MINUTE` | `0` | Minute for the nightly processing run |
| `WORKER_BATCH_TRIGGER_SIZE` | `50` | Pending log count to trigger immediate processing |
| `WORKER_BATCH_PROCESS_LIMIT` | `100` | Max logs to process in a single run |

### Fine-Tuning (Slow Loop)

| Variable | Default | Description |
|---|---|---|
| `FINE_TUNING_THRESHOLD` | `500` | Number of staged training pairs to trigger JSONL export |
| `FINE_TUNING_EXPORT_DIR` | `./exports` | Directory for exported `.jsonl` files |
| `FINE_TUNING_PROVIDER` | `local` | `local` (file only), `openai`, or `vllm` |
| `FINE_TUNING_REPLAY_BUFFER_RATIO` | `0.15` | Fraction of older data mixed in to prevent catastrophic forgetting |

### MCP Server (Fast Loop)

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport layer: `stdio` or `sse` |
| `MCP_SSE_HOST` | `0.0.0.0` | SSE server bind address (when `MCP_TRANSPORT=sse`) |
| `MCP_SSE_PORT` | `8080` | SSE server port |
| `MCP_SEARCH_TOP_K` | `5` | Max number of rules returned per search |
| `MCP_SEARCH_MIN_SCORE` | `0.5` | Minimum cosine similarity to include a rule |

### Embedding Model

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Sentence-transformer model for embeddings |
| `EMBEDDING_DIMENSION` | `1024` | Vector dimension (must match the model) |

### Database & Redis

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `nightshift` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `nightshift` | PostgreSQL password |
| `POSTGRES_DB` | `nightshift` | Database name |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |

---

## Running the System

The system has four services that run independently. Each should be started in its own terminal.

### 1. API Server (Capture Layer)

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Accepts interaction payloads at `http://localhost:8000/api/logs`.

### 2. Celery Worker (Night-Shift Agent)

```bash
celery -A app.worker.celery_app worker --loglevel=info
```

Processes pending interaction logs by sending them to the configured LLM, extracting rules, generating embeddings, and staging training pairs.

### 3. Celery Beat Scheduler

```bash
celery -A app.worker.celery_app beat --loglevel=info
```

Fires the nightly cron trigger and periodic fine-tuning threshold checks.

### 4. MCP Server (Fast Loop)

```bash
python -m app.mcp.server
```

Makes the `search_active_rules` tool available to AI agents.

---

## API Reference

### `GET /health`

Health check endpoint.

| Field | Value |
|---|---|
| **Method** | `GET` |
| **URL** | `/health` |
| **Auth** | None |
| **Response** | `{"status": "ok", "service": "nightshift-api"}` |

---

### `POST /api/logs`

Ingest a raw interaction log. This is the primary endpoint your application calls after a user edits an AI-generated output.

| Field | Value |
|---|---|
| **Method** | `POST` |
| **URL** | `/api/logs` |
| **Content-Type** | `application/json` |
| **Success Code** | `201 Created` |

#### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `system_prompt` | `string` | ✅ | The system instructions the AI was operating under |
| `user_input` | `string` | ✅ | The user's request or uploaded source text |
| `ai_output` | `string` | ✅ | The exact AI-generated response (before any edits) |
| `human_correction` | `string` | ✅ | The final text after the user manually edited the AI output |

#### Example Request

```bash
curl -X POST http://localhost:8000/api/logs \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a legal drafting assistant specialising in biotech licensing agreements.",
    "user_input": "Draft an indemnification clause for a license agreement between Acme Corp and BioGen Ltd.",
    "ai_output": "The Licensee shall indemnify the Licensor against all claims arising from use of the licensed technology.",
    "human_correction": "The Licensee shall indemnify and hold harmless the Licensor against all direct claims arising from use of the Licensed Technology, excluding any indirect, consequential, or punitive damages."
  }'
```

#### Example Response

```json
{
  "log_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "timestamp": "2026-03-21T20:05:00Z",
  "message": "Interaction log captured successfully."
}
```

#### Error Responses

| Code | Condition |
|---|---|
| `422 Unprocessable Entity` | Missing or empty required field |
| `500 Internal Server Error` | Database connection failure |

---

### MCP Tool: `search_active_rules`

Available via the Night-Shift MCP server. AI agents call this tool before generating text to retrieve relevant user preferences.

| Field | Value |
|---|---|
| **Tool Name** | `search_active_rules` |
| **Transport** | `stdio` (default) or `sse` |
| **Input** | `{"query": "description of the current drafting task"}` |
| **Output** | JSON array of matching rules with similarity scores |

#### Example Input

```json
{
  "query": "drafting an indemnification clause for a biotech license agreement"
}
```

#### Example Output

```json
[
  {
    "rule_id": "f8a1b2c3-d4e5-6789-0abc-def123456789",
    "rule_summary": "Always exclude indirect, consequential, and punitive damages from indemnification clauses.",
    "score": 0.8721
  },
  {
    "rule_id": "b9c2d3e4-f5a6-7890-1bcd-ef2345678901",
    "rule_summary": "Capitalise all defined terms (e.g., 'Licensed Technology', 'Confidential Information').",
    "score": 0.7534
  }
]
```

---

## Integrating into an Agentic Workflow

Night-Shift is designed to slot into any existing AI agent pipeline. Below is a step-by-step guide for integrating both the Capture Layer (input) and the Fast Loop (output).

### Step 1 — Instrument Your Application to Capture Edits

Wherever your application allows a user to edit AI-generated text, add a hook that fires when the user saves or accepts the final version. Send the four required fields to the Night-Shift API.

**Python Example:**

```python
import httpx

NIGHTSHIFT_URL = "http://localhost:8000"

async def capture_user_edit(
    system_prompt: str,
    user_input: str,
    ai_output: str,
    human_correction: str,
) -> dict:
    """
    Send the user's correction to Night-Shift for processing.
    Call this when the user clicks 'Save' or 'Accept' after editing.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NIGHTSHIFT_URL}/api/logs",
            json={
                "system_prompt": system_prompt,
                "user_input": user_input,
                "ai_output": ai_output,
                "human_correction": human_correction,
            },
        )
        response.raise_for_status()
        return response.json()
```

**JavaScript / TypeScript Example:**

```javascript
const NIGHTSHIFT_URL = "http://localhost:8000";

async function captureUserEdit(systemPrompt, userInput, aiOutput, humanCorrection) {
  const response = await fetch(`${NIGHTSHIFT_URL}/api/logs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      system_prompt: systemPrompt,
      user_input: userInput,
      ai_output: aiOutput,
      human_correction: humanCorrection,
    }),
  });

  if (!response.ok) throw new Error(`Night-Shift API error: ${response.status}`);
  return await response.json();
}
```

### Step 2 — Connect Your AI Agent to the MCP Server (Fast Loop)

The MCP server exposes the `search_active_rules` tool. Configure your AI agent to connect to it and call the tool before generating any text.

**Option A: stdio transport (same machine)**

Add the Night-Shift MCP server to your agent's MCP configuration:

```json
{
  "mcpServers": {
    "nightshift": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/NightShift"
    }
  }
}
```

**Option B: SSE transport (network-accessible)**

Set `MCP_TRANSPORT=sse` in `.env`, start the MCP server, and point your agent to:

```
http://<nightshift-host>:8080/sse
```

### Step 3 — Use the Rules in Your Agent's Pre-Flight

Before your agent generates a response, query Night-Shift for relevant preferences and inject them into the system prompt:

```python
# Inside your agent's generation pipeline
rules = await mcp_client.call_tool(
    "search_active_rules",
    {"query": "drafting a milestone payment clause for a pharma license"}
)

# Inject the rules into the agent's context
if rules:
    preference_block = "\n".join(
        f"- {r['rule_summary']}" for r in rules
    )
    system_prompt += (
        f"\n\n## User Preferences (from Night-Shift)\n"
        f"Apply these rules to your output:\n{preference_block}"
    )
```

### Step 4 — Let the Slow Loop Handle Fine-Tuning

No integration required. When the number of staged training pairs reaches the configured `FINE_TUNING_THRESHOLD` (default: 500), Night-Shift automatically:

1. Mixes the new pairs with 15% historical data (preventing catastrophic forgetting).
2. Exports a timestamped `.jsonl` file to the `exports/` directory.
3. (Future) Triggers the fine-tuning API if `FINE_TUNING_PROVIDER` is set to `openai` or `vllm`.

You can then use the exported file to fine-tune your model with your preferred provider.

### Integration Architecture Diagram

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      YOUR AGENTIC APPLICATION                      │
  │                                                                    │
  │  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
  │  │  User edits   │    │  Agent generates  │    │  Agent queries   │  │
  │  │  AI output    │───►│  with preferences │◄───│  Night-Shift MCP │  │
  │  └──────┬───────┘    └──────────────────┘    └────────┬─────────┘  │
  │         │                                             │            │
  └─────────┼─────────────────────────────────────────────┼────────────┘
            │                                             │
            ▼                                             ▼
  ┌──────────────────┐                         ┌──────────────────────┐
  │  POST /api/logs  │                         │  MCP: search_active  │
  │  (Capture Layer) │                         │  _rules (Fast Loop)  │
  └────────┬─────────┘                         └──────────┬───────────┘
           │                                              │
           ▼                                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                        NIGHT-SHIFT SYSTEM                       │
  │                                                                  │
  │   Raw Logs  ──►  LLM Agent  ──►  Rules (vectors) + Pairs        │
  │                                       │              │           │
  │                                  Fast Loop      Slow Loop        │
  │                                  (MCP search)   (JSONL export)   │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
NightShift/
├── app/
│   ├── core/              # Config, logging, embedding utilities
│   │   ├── config.py      # All settings loaded from .env
│   │   ├── logging.py     # Structured logging (JSON / console)
│   │   └── embeddings.py  # BGE-M3 embedding generation
│   ├── db/                # Database layer
│   │   ├── models.py      # SQLAlchemy ORM models (3 tables)
│   │   └── session.py     # Async engine + session factory
│   ├── api/               # FastAPI capture layer
│   │   ├── main.py        # App entry point + health check
│   │   └── routes/
│   │       └── ingestion.py  # POST /api/logs
│   ├── worker/            # Celery background processing
│   │   ├── celery_app.py  # Celery config + beat schedules
│   │   ├── tasks.py       # Task definitions (cron + batch)
│   │   ├── agent.py       # Processing orchestrator
│   │   ├── llm_client.py  # Multi-provider LLM client
│   │   └── processor.py   # JSON parsing + DB persistence
│   ├── mcp/               # MCP server (Fast Loop)
│   │   ├── server.py      # stdio / SSE transport
│   │   └── tools.py       # search_active_rules tool
│   └── finetune/          # Slow Loop
│       ├── monitor.py     # Threshold checker
│       └── exporter.py    # JSONL export + replay buffer
├── alembic/               # Database migrations
├── tests/                 # Pytest test suite
├── scripts/               # DB init scripts
├── docker-compose.yml     # Postgres + Redis
├── .env.example           # Configuration template
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_worker.py -v
```

The test suite covers:
- Configuration loading, defaults, and validation
- API endpoint payload validation and responses
- LLM output JSON parsing and error handling
- Fine-tuning threshold math and replay buffer calculations

---

## License

MIT
