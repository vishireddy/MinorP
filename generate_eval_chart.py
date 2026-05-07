import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Genuine metrics from the LLM-as-a-Judge eval run ──────────────
labels = ['Base LLM\n(No RAG)', 'Naive RAG', 'Adaptive RAG\n(LexPulse)']
overall  = [54.9, 54.9, 70.6]
tricky   = [55.6, 44.4, 72.2]

x = np.arange(len(labels))
width = 0.32

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.8,
    'axes.facecolor': '#f9fafb',
    'figure.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(11, 6.5))

# ── Colors ─────────────────────────────────────────────────────────
COLOR_NORAG  = '#94a3b8'   # slate
COLOR_NAIVE  = '#60a5fa'   # blue
COLOR_AWARE  = '#10b981'   # teal (brand color)

bars1 = ax.bar(x - width/2, overall, width,
               label='Overall Accuracy (%)',
               color=[COLOR_NORAG, COLOR_NAIVE, COLOR_AWARE],
               alpha=0.9, edgecolor='white', linewidth=1.5,
               zorder=3)

bars2 = ax.bar(x + width/2, tricky, width,
               label='Amendment-Trap Accuracy (%)',
               color=[COLOR_NORAG, COLOR_NAIVE, COLOR_AWARE],
               alpha=0.55, edgecolor='white', linewidth=1.5,
               hatch='//', zorder=3)

# ── Value labels ───────────────────────────────────────────────────
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#1e293b')

for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{bar.get_height():.1f}%',
            ha='center', va='bottom',
            fontsize=10, fontweight='semibold', color='#475569')

# ── Annotation: improvement arrow for Aware RAG ────────────────────
ax.annotate('+15.7pp\nvs Naive RAG',
            xy=(x[2] - width/2, 70.6),
            xytext=(x[2] - width/2 + 0.45, 76),
            fontsize=9, color='#059669', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5))

# ── Axes ───────────────────────────────────────────────────────────
ax.set_ylim(0, 90)
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#374151')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold', color='#1e293b')
ax.yaxis.set_tick_params(labelsize=10)

# ── Title ──────────────────────────────────────────────────────────
ax.set_title('LexPulse Evaluation: RAG Pipeline Comparison\n'
             'LLM-as-a-Judge · 51 Indian Legal Questions · Groq LLaMA-3.1-8b',
             fontsize=13, fontweight='bold', color='#0f172a', pad=18)

# ── Legend ─────────────────────────────────────────────────────────
solid  = mpatches.Patch(color='#64748b', alpha=0.9, label='Overall Accuracy (51 Questions)')
hatched = mpatches.Patch(facecolor='#64748b', alpha=0.55, hatch='//', label='Amendment-Trap Accuracy (18 Tricky Queries)')
ax.legend(handles=[solid, hatched], fontsize=10, framealpha=0.9,
          loc='upper left', edgecolor='#e2e8f0')

plt.tight_layout(pad=1.5)
plt.savefig('data/lexpulse_eval_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved to data/lexpulse_eval_comparison.png")
