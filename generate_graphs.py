import json
import matplotlib.pyplot as plt
import numpy as np

# Data metrics from your eval_results_v2.json
labels = ['Base LLM\n(No RAG)', 'Naive RAG', 'Adaptive RAG\n(Proposed)']
overall_accuracy = [76.47, 62.74, 66.66]
tricky_accuracy = [72.22, 44.44, 61.11]

x = np.arange(len(labels))
width = 0.35

# Set up the plot style for a professional academic paper
plt.style.use('bmh')
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, overall_accuracy, width, label='Overall Accuracy', color='#2C3E50', edgecolor='black')
rects2 = ax.bar(x + width/2, tricky_accuracy, width, label='Tricky Subset (Amendments)', color='#E74C3C', edgecolor='black')

# Add labels, title, and formatting
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: Standard vs. Relationship-Aware RAG', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='upper right')

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('data/accuracy_comparison_chart.png', dpi=600, bbox_inches='tight')
print("Graph successfully saved to data/accuracy_comparison_chart.png!")
