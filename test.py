import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. Données spaCy + CamemBERT
# ---------------------------
models = [
    # spaCy
    ("spaCy", "ner_blank_5000", 69.38, 95.98, 9.5),
    ("spaCy", "ner_blank_full", 69.29, 95.92, 5.4),
    ("spaCy", "ner_blank_3000", 70.00, 95.86, 5.2),
    ("spaCy", "ner_blank_1000", 67.72, 94.92, 5.3),
    ("spaCy", "ner_md_full", 54.94, 90.55, 13.1),
    ("spaCy", "ner_lg_full", 55.03, 90.26, 8.5),
    ("spaCy", "ner_sm_full", 52.32, 86.93, 11.9),
    # CamemBERT - tailles
    ("CamemBERT", "camembert_ner_3000", 75.42, 98.03, 108),
    ("CamemBERT", "camembert_ner_5000", 72.88, 97.34, 112),
    ("CamemBERT", "camembert_ner_full", 72.00, 97.06, 96),
    ("CamemBERT", "camembert_ner_1000", 71.48, 96.53, 161),
    # CamemBERT - LR
    ("CamemBERT", "camembert_lr1e-05", 75.07, 98.00, 78),
    ("CamemBERT", "camembert_lr3e-05", 72.18, 96.88, 91),
    ("CamemBERT", "camembert_lr5e-05", 71.92, 97.03, 80),
    # CamemBERT - batch
    ("CamemBERT", "camembert_bs32", 72.35, 97.14, 80),
    ("CamemBERT", "camembert_bs16", 71.83, 96.91, 81),
    ("CamemBERT", "camembert_bs8", 70.87, 96.65, 78),
    # CamemBERT - scheduler
    ("CamemBERT", "camembert_sched_cosine", 71.74, 96.88, 134),
    ("CamemBERT", "camembert_sched_linear", 71.13, 96.65, 135),
]

families = [m[0] for m in models]
names = [m[1] for m in models]
accuracy = [m[2] for m in models]
f1 = [m[3] for m in models]
latency = [m[4] for m in models]

indices = np.arange(len(models))

# ---------------------------
# 2. Bar chart des F1 (spaCy vs CamemBERT)
# ---------------------------
plt.figure(figsize=(14, 6))

# couleurs différentes par famille
colors = ["tab:blue" if fam == "spaCy" else "tab:orange" for fam in families]

plt.bar(indices, f1, color=colors)

# Mettre en avant les modèles recommandés
for i, name in enumerate(names):
    if name == "ner_blank_3000" or name == "camembert_ner_3000":
        plt.bar(i, f1[i], color="tab:green")

plt.xticks(indices, names, rotation=90)
plt.ylabel("F1 Score (%)")
plt.title("Comparaison F1 spaCy vs CamemBERT (toutes variantes)")
plt.grid(axis="y", alpha=0.3)

# Légende manuelle
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="tab:blue", label="spaCy"),
    Patch(color="tab:orange", label="CamemBERT"),
    Patch(color="tab:green", label="Modèles recommandés"),
]
plt.legend(handles=legend_handles, loc="lower right")

plt.tight_layout()
plt.savefig("camembert_spacy_f1_comparison.png", dpi=300)

# ---------------------------
# 3. Bar chart des temps d'inférence
# ---------------------------
plt.figure(figsize=(14, 6))

plt.bar(indices, latency, color=colors)
for i, name in enumerate(names):
    if name == "ner_blank_3000" or name == "camembert_ner_3000":
        plt.bar(i, latency[i], color="tab:green")

plt.xticks(indices, names, rotation=90)
plt.ylabel("Temps d'inférence (ms)")
plt.title("Comparaison des temps d'inférence spaCy vs CamemBERT")
plt.grid(axis="y", alpha=0.3)
plt.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
plt.savefig("camembert_spacy_latency_comparison.png", dpi=300)

plt.show()
