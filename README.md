💊 Formulary Assistant: LLM Semantic Caching LayerFormulary Assistant is a production-ready LLM routing pipeline that significantly optimizes response times and drastically cuts operational costs for openFDA medical formulary queries. By wrapping the core LLM in a dual Semantic and Fuzzy Caching Layer powered by Redis, the system intercepts repetitive or meaning-identical user queries—yielding up to a 40x speed improvement and a 75% reduction in API overhead.🚀 Key FeaturesMulti-Layered Redis Cache: Intercepts queries using both exact keyword-fuzzy logic and deep semantic vector search.Smart Thresholding: Captures queries sharing the same intent (e.g., typos or altered sentence structure) using verified cosine similarity thresholds.Automated Seed Embedding Generation: Dynamically builds vector caches using HuggingFace Hub embeddings for 200 real openFDA dataset entries on startup.Seamless LLM Fallback: Gracefully degrades to a paid OpenAI route if similarity fallback criteria are not met, caching the response for subsequent queries.Telemetry & Cost Tracking: Emits instant granular metrics tracking cache hits, token usage, financial expenditure, and millisecond latency.🛠️ System Architecture User Query
     │
     ▼
┌──────────────┐      [Match found]      ┌──────────────────────┐
│  Redis Cache │ ──────────────────────> │ Return Stored Answer │
└──────────────┘                         └──────────────────────┘
     │
     │ [Cache Miss / Low Similarity]
     ▼
┌──────────────┐                         ┌──────────────────────┐
│  OpenAI API  │ ──────────────────────> │ Cache & Serve User   │
└──────────────┘                         └──────────────────────┘
📋 Environment ConfigurationThe application requires specific runtime configurations to control testing constraints and target evaluation sizes. Configure your environment before execution:DEMO_QUERIES_FIRST4: Set to "1" to force a benchmark run across 4 un-labeled demo query cases.MAX_SEED_ROWS: The total number of openFDA raw medical entries to embed and push into the local Redis storage cache layer.HF_TOKEN: (Optional) Your Hugging Face Hub token to gain access to higher rate limits and expedited embedding generation downloads.💻 Quick Start1. PrerequisitesEnsure you have Python 3.10+ installed and a running instance of Redis Server locally or via Docker.bashdocker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
Use code with caution.2. Run the BenchmarksActivate your virtual environment and initialize the pipeline application script:powershell# Windows PowerShell Execution
$env:DEMO_QUERIES_FIRST4="1"
$env:MAX_SEED_ROWS="200"
python.exe -m src.app
Use code with caution.📊 Evaluation Outputs & Benchmark ReportWhen the benchmark evaluation completes, the program cleans stale cache partitions, indexes your vector spaces, and dumps real-time operational metrics:Query Routing PerformancetextCase 1
Query : Can you give me a substitute option for basic care daytime severe cold and flu because it's non-covered?
Cache : HIT (Semantic Cache Hit (Verified Score: 4.60))
Cache latency (ms): 97.82
----------------------------------------------------------------------------------------------------
Case 2
Query : Prior authorization criteria for Ozempic 1mg injection?
Cache : MISS (Cache Miss (Low Similarity: 0.39)) -> LLM
Cache latency (ms): 47.67
LLM latency (ms): 2197.77
LLM tokens used: 145
...
Use code with caution.Operational Performance Performance SummaryMetricTarget Baseline PerformanceTotal Cache Hits (Blocked Locally)3 RequestsTotal LLM Misses (Pushed to OpenAI)1 RequestAverage Cache Layer Latency54.40 msAverage LLM Remote Latency2,197.77 msTotal Pipeline Financial Overhead$5.91e-05 USD📂 Project Directory StructuretextFormulary_Assistant_Project_caching/
├── .venv-1/                 # Python Local Virtual Environment
├── src/
│   ├── app.py               # Application entryway & pipeline runner
│   ├── cache/               # Redis Semantic & Fuzzy index client models
│   ├── embeddings/          # HuggingFace transformer pipeline wrapper
│   └── llm/                 # OpenAI endpoint fallback router
└── README.md                # Documentation repository overview
Use code with caution.If you want to append the specific name of the embedding model or add an installation instructions block for pip install dependencies, let me know!
