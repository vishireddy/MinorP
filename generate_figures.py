"""
generate_figures.py
────────────────────
Generates publication-quality evaluation charts from saved results.
Run: python generate_figures.py
Outputs saved to data/figures/
"""

import os
import json
import sys

os.makedirs("data/figures", exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# ── Style ─────────────────────────────────────────────────────
COLORS = {
    "naive_llm": "#EF4444",   # red
    "naive_rag": "#F59E0B",   # amber
    "aware_rag": "#10B981",   # green
    "ablation":  "#6366F1",   # indigo
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def load_eval():
    path = "data/eval_results.json"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Run the Full Evaluation in the app first.")
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload.get("results", payload)


def load_ablation():
    path = "data/ablation_results.json"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Run the Ablation Study in the app first.")
        return None
    with open(path) as f:
        return json.load(f)


def load_ragas():
    path = "data/ragas_results.json"
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Run RAGAS Analysis in the app first.")
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload.get("results", payload)


# ── Figure 1: Category Accuracy Bar Chart ─────────────────────
def plot_category_accuracy(results):
    cats = list(results["category_scores"].keys())
    nl = [results["category_scores"][c]["naive_llm_pass"] / results["category_scores"][c]["total"] * 100 for c in cats]
    nr = [results["category_scores"][c]["naive_rag_pass"] / results["category_scores"][c]["total"] * 100 for c in cats]
    aw = [results["category_scores"][c]["aware_pass"] / results["category_scores"][c]["total"] * 100 for c in cats]

    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, nl, w, label="Gemma2-9b (No RAG)", color=COLORS["naive_llm"], alpha=0.9)
    ax.bar(x,     nr, w, label="Mixtral (Naive RAG)", color=COLORS["naive_rag"], alpha=0.9)
    ax.bar(x + w, aw, w, label="LLaMA3 (Aware RAG)", color=COLORS["aware_rag"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title("Accuracy by Legal Category — LLM-as-a-Judge Evaluation (50 Questions)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)

    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=2)

    plt.tight_layout()
    out = "data/figures/category_accuracy.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Figure 2: Overall Pass Rate Comparison ────────────────────
def plot_overall_comparison(results):
    m = results["metrics"]
    models = ["Gemma2-9b\n(No RAG)", "Mixtral-8x7b\n(Naive RAG)", "LLaMA3.1-8b\n(Aware RAG)"]
    pass_rates = [m["naive_llm_accuracy"], m["naive_rag_accuracy"], m["aware_accuracy"]]
    avg_scores = [m["naive_llm_avg_score"], m["naive_rag_avg_score"], m["aware_avg_score"]]
    colors = [COLORS["naive_llm"], COLORS["naive_rag"], COLORS["aware_rag"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(models, pass_rates, color=colors, alpha=0.9, width=0.5)
    ax1.set_ylabel("Pass Rate (%)", fontsize=11)
    ax1.set_title("Overall Pass Rate (Score ≥ 6/10)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.bar_label(bars1, fmt="%.1f%%", fontsize=11, padding=4)

    bars2 = ax2.bar(models, avg_scores, color=colors, alpha=0.9, width=0.5)
    ax2.set_ylabel("Avg Judge Score (0–10)", fontsize=11)
    ax2.set_title("Average Judge Score (out of 10)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 12)
    ax2.bar_label(bars2, fmt="%.2f", fontsize=11, padding=4)

    plt.suptitle("LLM-as-a-Judge Comparative Evaluation — 50 Questions, 16 Legal Categories",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = "data/figures/overall_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Figure 3: Amendment-Trap Performance ──────────────────────
def plot_amendment_trap(results):
    m = results["metrics"]
    models = ["Gemma2-9b\n(No RAG)", "Mixtral-8x7b\n(Naive RAG)", "LLaMA3.1-8b\n(Aware RAG)"]
    trap_rates = [m["tricky_naive_llm_accuracy"], m["tricky_naive_rag_accuracy"], m["tricky_aware_accuracy"]]
    overall_rates = [m["naive_llm_accuracy"], m["naive_rag_accuracy"], m["aware_accuracy"]]
    colors = [COLORS["naive_llm"], COLORS["naive_rag"], COLORS["aware_rag"]]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, overall_rates, w, label="All Questions", color=colors, alpha=0.5, edgecolor="none")
    ax.bar(x + w/2, trap_rates,   w, label="Amendment-Trap Only", color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title(f"Amendment-Trap Question Performance ({m['tricky_total']} targeted questions)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)

    handles = [
        mpatches.Patch(color="gray", alpha=0.5, label="All 50 Questions"),
        mpatches.Patch(color="gray", alpha=0.9, label="Amendment-Trap Questions", linewidth=0.8),
    ]
    ax.legend(handles=handles, fontsize=10)
    plt.tight_layout()
    out = "data/figures/amendment_trap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Figure 4: Ablation Study (4 Pipelines) ───────────────────
def plot_ablation(ablation):
    am = ablation["metrics"]

    pipelines = [
        "P1: No RAG\n(Gemma2-9b)",
        "P2: Naive RAG\n(Mixtral)",
        "P3: Hybrid RAG\n(LLaMA3)",
        "P4: Aware RAG\n(LLaMA3) ★",
    ]
    pass_rates  = [am["no_rag_pass_rate"],  am["naive_rag_pass_rate"],
                   am["hybrid_pass_rate"],   am["aware_pass_rate"]]
    avg_scores  = [am["no_rag_avg_score"],   am["naive_rag_avg_score"],
                   am["hybrid_avg_score"],   am["aware_avg_score"]]
    trap_rates  = [am["tricky_no_rag"],      am["tricky_naive_rag"],
                   am["tricky_hybrid"],      am["tricky_aware"]]
    colors = [COLORS["naive_llm"], COLORS["naive_rag"], "#6366F1", COLORS["aware_rag"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 4a: Pass rate with component-gain annotations
    bars = axes[0].bar(pipelines, pass_rates, color=colors, alpha=0.9, width=0.55)
    axes[0].set_title("Overall Pass Rate (%)\n(Score ≥ 6/10)", fontweight="bold")
    axes[0].set_ylim(0, 120)
    axes[0].bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    gains = [
        ("", None),
        (f"+{am['retrieval_gain']:.1f}%\nretrieval", 1),
        (f"+{am['hybrid_gain']:.1f}%\nhybrid",     2),
        (f"+{am['graph_gain']:.1f}%\ngraph",       3),
    ]
    for label, idx in [(g, x) for g, x in gains if x is not None]:
        axes[0].text(idx, pass_rates[idx] + 7, label, ha="center",
                     fontsize=8, color="black", fontweight="bold")

    # 4b: Avg score
    bars2 = axes[1].bar(pipelines, avg_scores, color=colors, alpha=0.9, width=0.55)
    axes[1].set_title("Avg Judge Score (/10)", fontweight="bold")
    axes[1].set_ylim(0, 12)
    axes[1].bar_label(bars2, fmt="%.2f", padding=4, fontsize=10)

    # 4c: Amendment-trap
    bars3 = axes[2].bar(pipelines, trap_rates, color=colors, alpha=0.9, width=0.55)
    axes[2].set_title(f"Amendment-Trap Pass Rate\n({am['tricky_total']} questions)", fontweight="bold")
    axes[2].set_ylim(0, 120)
    axes[2].bar_label(bars3, fmt="%.1f%%", padding=4, fontsize=10)

    plt.suptitle(
        "Ablation Study: Progressive Component Contribution\n"
        "P1→P2: +Retrieval   P2→P3: +Hybrid(BM25)   P3→P4: +Relationship Graph",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out = "data/figures/ablation_study.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Figure 5: RAGAS Metrics Comparison ────────────────────────
def plot_ragas(ragas):
    metrics = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall", "RAGAS\nScore"]
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "ragas_score"]
    nr = ragas["naive_rag"]
    aw = ragas["aware_rag"]
    naive_vals = [nr[k] for k in keys]
    aware_vals = [aw[k] for k in keys]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, naive_vals, w, label="Naive RAG (Mixtral)", color=COLORS["naive_rag"], alpha=0.9)
    bars2 = ax.bar(x + w/2, aware_vals, w, label="Aware RAG (LLaMA3)", color=COLORS["aware_rag"], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax.set_title(f"RAGAS Metrics: Naive RAG vs Aware RAG ({ragas['n_questions']} Questions)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    out = "data/figures/ragas_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Figure 6: Latency Comparison ─────────────────────────────
def plot_latency(results):
    m = results["metrics"]
    nl_lat = m.get("naive_llm_avg_latency_ms")
    nr_lat = m.get("naive_rag_avg_latency_ms")
    aw_lat = m.get("aware_avg_latency_ms")
    if not (nl_lat and nr_lat and aw_lat):
        print("[SKIP] Latency data not available (run evaluation with updated code).")
        return None

    models = ["Gemma2-9b\n(No RAG)", "Mixtral-8x7b\n(Naive RAG)", "LLaMA3.1-8b\n(Aware RAG)"]
    lats = [nl_lat, nr_lat, aw_lat]
    colors = [COLORS["naive_llm"], COLORS["naive_rag"], COLORS["aware_rag"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, lats, color=colors, alpha=0.9, width=0.45)
    ax.set_ylabel("Avg Response Latency (ms)", fontsize=11)
    ax.set_title("Average Response Latency per Pipeline", fontsize=12, fontweight="bold")
    ax.bar_label(bars, fmt="%d ms", padding=4, fontsize=11)
    ax.set_ylim(0, max(lats) * 1.25)
    plt.tight_layout()
    out = "data/figures/latency_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out}")
    return out


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Publication Figure Generator")
    print("  Output: data/figures/")
    print("=" * 60)

    eval_results = load_eval()
    abl_results  = load_ablation()
    ragas_results = load_ragas()

    generated = []

    if eval_results:
        generated.append(plot_category_accuracy(eval_results))
        generated.append(plot_overall_comparison(eval_results))
        generated.append(plot_amendment_trap(eval_results))
        generated.append(plot_latency(eval_results))
    else:
        print("[SKIP] eval_results.json not found — skipping 3 figures.")

    if abl_results:
        generated.append(plot_ablation(abl_results))
    else:
        print("[SKIP] ablation_results.json not found — skipping ablation figure.")

    if ragas_results:
        generated.append(plot_ragas(ragas_results))
    else:
        print("[SKIP] ragas_results.json not found — skipping RAGAS figure.")

    generated = [g for g in generated if g]
    print(f"\n✅ Generated {len(generated)} figure(s):")
    for p in generated:
        print(f"   {p}")
    print("\nRun from project root: python generate_figures.py")
