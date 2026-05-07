import json
import matplotlib.pyplot as plt
import numpy as np

# Data metrics focusing ONLY on the RAG comparison
labels = ['Naive RAG\n(Standard Chunking)', 'Relationship-Aware RAG\n(Proposed Architecture)']
overall_accuracy = [62.74, 66.66]
tricky_accuracy = [44.44, 61.11]

x = np.arange(len(labels))
width = 0.35

# Set up the plot style for a professional academic paper
plt.style.use('bmh')
fig, ax = plt.subplots(figsize=(8, 6))

# Create bars
rects1 = ax.bar(x - width/2, overall_accuracy, width, label='Overall Accuracy', color='#34495E', edgecolor='black')
rects2 = ax.bar(x + width/2, tricky_accuracy, width, label='Tricky Subset (Amendments)', color='#2ECC71', edgecolor='black') # Using green to highlight the positive result

# Add labels, title, and formatting
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('RAG Architecture Comparison: Curing Amendment Blindness', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='upper left')

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('data/rag_only_comparison_chart.png', dpi=300, bbox_inches='tight')
print("Graph successfully saved to data/rag_only_comparison_chart.png!")
