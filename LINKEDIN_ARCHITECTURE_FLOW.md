# Formulary Assistant (Semantic Cache) — Architecture Flow

## 10-second summary
A CLI formulary assistant that routes each user query through a **3-layer cache** (fuzzy → Redis vector search → cross-encoder verification). If no safe match is found, it falls back to an **OpenAI LLM**. The app logs **hit/miss routing** plus **latency, tokens, and cost**.

## System diagram (Mermaid)
Paste this into a Mermaid viewer (or GitHub Markdown) and screenshot for LinkedIn.

```mermaid
flowchart TD
  %% Entry
  U[User / CLI] --> A[Run src/app.py]
  A --> E[Load .env + config]
  A --> D1[Load data/real_fda_cache.csv]
  A --> D2[Load validation queries\n(data/real_fda_validation.csv OR built-in demo Q1-Q4)]

  %% Seeding
  D1 --> S[Seed cache]
  S --> FZ[Layer 1: RapidFuzz dictionary\n(drug_key -> approved_response)]
  S --> VE[Layer 2: Bi-encoder embeddings\nSentenceTransformer all-MiniLM-L6-v2]
  VE --> R[Redis Vector Index (redisvl)\nCOSINE similarity + plan_id filter]

  %% Per-query routing
  D2 --> Q[For each query]
  Q --> L1{Fuzzy match\nscore >= cutoff?}
  L1 -- Yes --> HIT1[HIT\nReturn cached approved_response\n+ cache latency]

  L1 -- No --> L2[Embed query (bi-encoder)]
  L2 --> RS[Redis vector search\n(top-1 candidate)]
  RS --> SIM{bi-similarity >= threshold?}
  SIM -- No --> MISS1[MISS\nRoute to LLM]

  SIM -- Yes --> CE[Layer 3: Cross-encoder verify\nms-marco-MiniLM-L-6-v2]
  CE --> OK{cross-score >= threshold?}
  OK -- Yes --> HIT2[HIT\nReturn verified cached response\n+ cache latency]
  OK -- No --> MISS2[MISS\nRoute to LLM]

  %% LLM fallback
  MISS1 --> LLM[OpenAI chat.completions\n(gpt-4o-mini)]
  MISS2 --> LLM
  LLM --> OUT[Print result +\nLLM latency/tokens/cost]

  %% Metrics
  HIT1 --> M[Metrics aggregator]
  HIT2 --> M
  OUT --> M
  M --> P[Print Operational Performance Metrics\n(hits, misses, avg latencies, tokens, cost)]

  %% Data pipeline (offline)
  DP[src/data_pipeline.py] --> FDA[openFDA bulk NDC JSON\n(data/drug-ndc-0001-of-0001.json)]
  FDA --> GEN[Generate cache + validation CSVs]
  GEN --> D1
  GEN --> D2
```

## What each module does
- `src/data_pipeline.py`
  - Parses the openFDA bulk NDC JSON and generates:
    - `data/real_fda_cache.csv` (seed rows)
    - `data/real_fda_validation.csv` (hit/miss evaluation traffic)
- `src/cache_engine.py` (`AdvancedCache`)
  - Layer 1: RapidFuzz typo-tolerant matching against `drug_key`
  - Layer 2: Bi-encoder embeddings + Redis vector search via `redisvl`
  - Layer 3: Cross-encoder verification to reduce false positives (dosage/nuance safety)
- `src/llm_agent.py` (`OpenAIFallbackAgent`)
  - Calls OpenAI when the cache path is not confident
  - Tracks token usage, estimated cost, and latency
- `src/app.py`
  - Orchestrates end-to-end run: seed → evaluate queries → print per-case routing + summary metrics

## LinkedIn caption (copy/paste)
Built a semantic caching router for formulary-style Q&A.

Flow:
1) Fuzzy match (typos)
2) Redis vector search (semantic similarity)
3) Cross-encoder verification (precision / dosage nuance)
4) If not confident → LLM fallback

Each query prints HIT/MISS + latency, and overall metrics include tokens + estimated cost.

#python #redis #vectorsearch #nlp #llm #caching #healthtech
