import os
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

try:
    # Preferred when running as a module: `python -m src.app`
    from src.cache_engine import AdvancedCache
    from src.llm_agent import OpenAIFallbackAgent
except ImportError:  # pragma: no cover
    # Fallback when running as a script: `python src/app.py`
    from cache_engine import AdvancedCache
    from llm_agent import OpenAIFallbackAgent


def print_operational_performance_metrics(metrics_log: dict) -> None:
    print("\nOperational Performance Metrics:")
    print(f"Total Cache Hits Requests Blocked Locally (Saved Money) : {metrics_log['cache_hits']}")
    print(f"Total LLM Misses Requests Pushed to OpenAI (Paid Route): {metrics_log['llm_misses']}")
    print(f"Total Tokens Consumed: {metrics_log['total_tokens_consumed']}")
    print(f"Total Financial Cost (USD): ${metrics_log['total_financial_cost_usd']}")
    print(
        f"Average Cache Latency (ms): {metrics_log['total_cache_latency_ms'] / max(metrics_log['cache_hits'], 1):.2f} ms"
    )
    print(
        f"Average LLM Latency (ms): {metrics_log['total_llm_latency_ms'] / max(metrics_log['llm_misses'], 1):.2f} ms"
    )


def _build_demo_validation_df_first4(cache_df: pd.DataFrame) -> pd.DataFrame:
    # Mirrors the first 4 cases used in the demo flow.
    seeded_key = str(cache_df.dropna(subset=["drug_key"]).iloc[0]["drug_key"])

    q1 = "Can you give me a substitute option for basic care daytime severe cold and flu because it's non-covered?"
    dosage_miss_examples = [
        "Substitute drug for Extra Strength Antacid Ultra Max 1000mg?",
        "What is the covered alternative for Paxlovid 300mg?",
        "Prior authorization criteria for Ozempic 1mg injection?",
    ]
    q2 = dosage_miss_examples[2]
    q3 = "Alternative for Sidlenafil?"
    q4 = f"Need a noncovered substitute for {seeded_key} solution"

    return pd.DataFrame({"query": [q1, q2, q3, q4]})


def run_evaluation(
    cache_engine,
    llm_agent,
    cache_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    plan_id: str,
    *,
    print_per_query: bool = False,
):
    """Runs the end-to-end cache->LLM evaluation loop and prints summary metrics."""
    # Load the cache df via cache engine seeding method to populate both the Redis index and the local fuzzy dictionary.
    cache_engine.seed_semantic_cache(cache_df)

    has_ground_truth = "ground_truth" in validation_df.columns
    y_true, y_pred = [], []

    metrics_log = {
        "cache_hits": 0,
        "llm_misses": 0,
        "total_tokens_consumed": 0,
        "total_financial_cost_usd": 0.0,
        "total_cache_latency_ms": 0.0,
        "total_llm_latency_ms": 0.0,
    }

    for case_num, row in enumerate(validation_df.itertuples(index=False), start=1):
        query = str(getattr(row, "query"))
        ground_truth = str(getattr(row, "ground_truth")) if has_ground_truth else None

        cache_start = time.time()
        status, cache_response_value = cache_engine.evaluate_request(plan_id, query)
        cache_end = time.time()
        local_execution_latency = (cache_end - cache_start) * 1000

        if "Hit" in str(status):
            predicted = "Hit"
            metrics_log["cache_hits"] += 1
            metrics_log["total_cache_latency_ms"] += local_execution_latency
            if print_per_query:
                print("-" * 100)
                print(f"Case {case_num}")
                print(f"Query : {query}")
                print(f"Cache : HIT ({status})")
                print(f"Cache latency (ms): {local_execution_latency:.2f}")
        else:
            predicted = "Miss"
            llm_result = llm_agent.generate_formulary_fallback(query)
            metrics_log["llm_misses"] += 1
            llm_latency_ms = float(llm_result.get("latency_ms", 0.0) or 0.0)
            llm_tokens_used = int(llm_result.get("tokens_used", 0) or 0)
            llm_cost_usd = float(llm_result.get("cost_usd", 0.0) or 0.0)
            metrics_log["total_llm_latency_ms"] += llm_latency_ms
            metrics_log["total_tokens_consumed"] += llm_tokens_used
            metrics_log["total_financial_cost_usd"] += llm_cost_usd
            if print_per_query:
                print("-" * 100)
                print(f"Case {case_num}")
                print(f"Query : {query}")
                print(f"Cache : MISS ({status}) -> LLM")
                print(f"Cache latency (ms): {local_execution_latency:.2f}")
                print(f"LLM latency (ms): {llm_latency_ms:.2f}")
                print(f"LLM tokens used: {llm_tokens_used}")
                print(f"LLM cost (USD): ${llm_cost_usd:.6f}")

        if has_ground_truth and ground_truth is not None:
            y_true.append(ground_truth)
            y_pred.append(predicted)

    if has_ground_truth:
        labels = ["Hit", "Miss"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels)

        print("Confusion Matrix:")
        print(cm)
        print("=" * 50)
        print("\nClassification Report:")
        print(report)
        print("=" * 50)

    print_operational_performance_metrics(metrics_log)

    return metrics_log


def main():
    
    cache_engine = AdvancedCache()
    llm_agent = OpenAIFallbackAgent()
    
    cache_df = pd.read_csv(os.path.join("data", "real_fda_cache.csv"))

    max_seed_rows = int(os.getenv("MAX_SEED_ROWS", "0"))
    if max_seed_rows > 0:
        cache_df = cache_df.head(max_seed_rows)

    demo_first4 = str(os.getenv("DEMO_QUERIES_FIRST4", "0")).strip('"') == "1"
    if demo_first4:
        validation_df = _build_demo_validation_df_first4(cache_df)
        print("DEMO_QUERIES_FIRST4=1 -> running evaluation on 4 demo queries (no ground_truth labels).")
    else:
        validation_df = pd.read_csv(os.path.join("data", "real_fda_validation.csv"))

    max_validation_rows = int(os.getenv("MAX_VALIDATION_ROWS", "0"))
    if (not demo_first4) and max_validation_rows > 0:
        validation_df = validation_df.head(max_validation_rows)

    print(
        f"Using limits: MAX_SEED_ROWS={max_seed_rows}, MAX_VALIDATION_ROWS={max_validation_rows} | "
        f"seed_rows={len(cache_df)}, validation_rows={len(validation_df)}"
    )

    print("=" * 100)
    plan_id = os.getenv("PLAN_ID", "CHOICE_DUMMY_PLAN")
    run_evaluation(cache_engine, llm_agent, cache_df, validation_df, plan_id, print_per_query=demo_first4)


if __name__ == "__main__":
    main()
    




