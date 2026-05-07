"""
results_manager.py
──────────────────
Handles persistent storage and export of evaluation results.
Enables research reproducibility and paper-ready statistics.
"""

import os
import json
import time
import io
from typing import Optional

EVAL_RESULTS_PATH = "data/eval_results_v2.json"
ABLATION_RESULTS_PATH = "data/ablation_results_v2.json"
RAGAS_RESULTS_PATH = "data/ragas_results_v2.json"


# ── SAVE / LOAD ────────────────────────────────────────────────

def save_eval_results(results: dict) -> None:
    """Persist LLM-as-Judge evaluation results to disk."""
    os.makedirs("data", exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "2.0",
        "results": results
    }
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ResultsManager] Saved eval results to {EVAL_RESULTS_PATH}")


def load_eval_results() -> Optional[dict]:
    """Load cached LLM-as-Judge results. Returns None if not found."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        return None
    try:
        with open(EVAL_RESULTS_PATH, "r") as f:
            payload = json.load(f)
        return payload
    except Exception as e:
        print(f"[ResultsManager] Failed to load eval results: {e}")
        return None


def save_ablation_results(results: dict) -> None:
    """Persist ablation study results to disk."""
    os.makedirs("data", exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results
    }
    with open(ABLATION_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ResultsManager] Saved ablation results to {ABLATION_RESULTS_PATH}")


def load_ablation_results() -> Optional[dict]:
    """Load cached ablation results. Returns None if not found."""
    if not os.path.exists(ABLATION_RESULTS_PATH):
        return None
    try:
        with open(ABLATION_RESULTS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ResultsManager] Failed to load ablation results: {e}")
        return None


def save_ragas_results(results: dict) -> None:
    """Persist RAGAS evaluation results to disk."""
    os.makedirs("data", exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results
    }
    with open(RAGAS_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ResultsManager] Saved RAGAS results to {RAGAS_RESULTS_PATH}")


def load_ragas_results() -> Optional[dict]:
    """Load cached RAGAS results. Returns None if not found."""
    if not os.path.exists(RAGAS_RESULTS_PATH):
        return None
    try:
        with open(RAGAS_RESULTS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ResultsManager] Failed to load RAGAS results: {e}")
        return None


# ── CSV EXPORT ─────────────────────────────────────────────────

def export_breakdown_to_csv_bytes(results: dict) -> bytes:
    """
    Converts the per-question breakdown to a CSV-formatted bytes object
    suitable for st.download_button().
    """
    import csv
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Q#", "Category", "Amendment_Trap", "Query",
        "NaiveLLM_Score", "NaiveLLM_Pass", "NaiveLLM_Reason",
        "NaiveRAG_Score", "NaiveRAG_Pass", "NaiveRAG_Reason",
        "AwareRAG_Score", "AwareRAG_Pass", "AwareRAG_Reason",
        "NaiveLLM_Latency_ms", "NaiveRAG_Latency_ms", "AwareRAG_Latency_ms",
        "Reference_Answer"
    ])

    breakdown = results.get("results", {}).get("breakdown", [])
    for i, r in enumerate(breakdown):
        writer.writerow([
            i + 1,
            r.get("category", ""),
            "YES" if r.get("tricky") else "NO",
            r.get("query", ""),
            r.get("naive_llm_score", 0),
            "PASS" if r.get("naive_llm_pass") else "FAIL",
            r.get("naive_llm_reason", ""),
            r.get("naive_rag_score", 0),
            "PASS" if r.get("naive_rag_pass") else "FAIL",
            r.get("naive_rag_reason", ""),
            r.get("aware_score", 0),
            "PASS" if r.get("aware_pass") else "FAIL",
            r.get("aware_reason", ""),
            r.get("naive_llm_latency_ms", "N/A"),
            r.get("naive_rag_latency_ms", "N/A"),
            r.get("aware_latency_ms", "N/A"),
            r.get("reference", ""),
        ])

    return output.getvalue().encode("utf-8")


def export_metrics_to_csv_bytes(results: dict) -> bytes:
    """Export top-level metrics summary to CSV."""
    import csv
    output = io.StringIO()
    writer = csv.writer(output)

    m = results.get("results", {}).get("metrics", {})
    writer.writerow(["Metric", "No RAG (LLaMA3.3-70b)", "Naive RAG (LLaMA3.3-70b)", "Aware RAG (LLaMA3.3-70b)"])
    writer.writerow(["Pass Rate (%)", f"{m.get('naive_llm_accuracy', 0):.1f}",
                     f"{m.get('naive_rag_accuracy', 0):.1f}", f"{m.get('aware_accuracy', 0):.1f}"])
    writer.writerow(["Avg Judge Score (/10)", f"{m.get('naive_llm_avg_score', 0):.2f}",
                     f"{m.get('naive_rag_avg_score', 0):.2f}", f"{m.get('aware_avg_score', 0):.2f}"])
    writer.writerow(["Amendment-Trap Pass Rate (%)", f"{m.get('tricky_naive_llm_accuracy', 0):.1f}",
                     f"{m.get('tricky_naive_rag_accuracy', 0):.1f}", f"{m.get('tricky_aware_accuracy', 0):.1f}"])
    writer.writerow(["Avg Latency (ms)", m.get('naive_llm_avg_latency_ms', 'N/A'),
                     m.get('naive_rag_avg_latency_ms', 'N/A'), m.get('aware_avg_latency_ms', 'N/A')])

    return output.getvalue().encode("utf-8")


# ── PAPER-READY STATS ──────────────────────────────────────────

def generate_paper_stats(results: dict) -> dict:
    """
    Extracts a clean stats dictionary from saved results for
    direct injection into LaTeX paper figures and tables.
    """
    r = results.get("results", {})
    m = r.get("metrics", {})

    stats = {
        "total_questions": m.get("total_queries", 50),
        "num_categories": len(r.get("category_scores", {})),
        "num_amendment_trap": m.get("tricky_total", 0),
        # Pass rates
        "naive_llm_pass_rate": round(m.get("naive_llm_accuracy", 0), 1),
        "naive_rag_pass_rate": round(m.get("naive_rag_accuracy", 0), 1),
        "aware_pass_rate": round(m.get("aware_accuracy", 0), 1),
        # Average scores
        "naive_llm_avg": round(m.get("naive_llm_avg_score", 0), 2),
        "naive_rag_avg": round(m.get("naive_rag_avg_score", 0), 2),
        "aware_avg": round(m.get("aware_avg_score", 0), 2),
        # Improvements
        "improvement_over_llm": round(m.get("rag_improvement_over_llm", 0), 1),
        "improvement_over_naive_rag": round(m.get("rag_improvement_over_naive_rag", 0), 1),
        # Amendment trap
        "trap_naive_llm": round(m.get("tricky_naive_llm_accuracy", 0), 1),
        "trap_naive_rag": round(m.get("tricky_naive_rag_accuracy", 0), 1),
        "trap_aware": round(m.get("tricky_aware_accuracy", 0), 1),
        # Latency
        "naive_llm_latency_ms": m.get("naive_llm_avg_latency_ms", "N/A"),
        "naive_rag_latency_ms": m.get("naive_rag_avg_latency_ms", "N/A"),
        "aware_latency_ms": m.get("aware_avg_latency_ms", "N/A"),
        # Timestamp
        "eval_timestamp": results.get("timestamp", "Unknown"),
    }
    return stats


def results_exist() -> bool:
    """Quick check if any cached results exist."""
    return os.path.exists(EVAL_RESULTS_PATH)


def ablation_results_exist() -> bool:
    """Quick check if any cached ablation results exist."""
    return os.path.exists(ABLATION_RESULTS_PATH)


def ragas_results_exist() -> bool:
    """Quick check if any cached RAGAS results exist."""
    return os.path.exists(RAGAS_RESULTS_PATH)
