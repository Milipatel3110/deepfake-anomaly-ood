"""
energy_hist.py
Visualize ΔEnergy and ROC performance from ODIN/energy-based OOD results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -------- CONFIG --------
CSV_PATH = "/home/STUDENTS/map0662/untag_project/outputs/ood_results_energy_calibrated_delta.csv"   # adjust if needed
SAVE_PREFIX = "../outputs/plots_energy_"                     # output prefix
DELTA_COLUMN = "delta_energy"                                # make sure this matches your column name

# -------- LOAD DATA --------
print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# If your file doesn't have delta_energy column yet:
if DELTA_COLUMN not in df.columns:
    print("ΔEnergy not found — computing manually from available columns...")
    if "energy_before" in df.columns and "energy_after" in df.columns:
        df[DELTA_COLUMN] = df["energy_after"] - df["energy_before"]
    else:
        raise ValueError("No ΔEnergy columns found in CSV!")

# labels: 0 = real, 1 = fake
y_true = df["label(fake=1)"].values
delta_E = df[DELTA_COLUMN].values

# -------- HISTOGRAM PLOT --------
plt.figure(figsize=(8,5))
plt.hist(delta_E[y_true == 0], bins=40, alpha=0.7, label="Real", color="#0077b6")
plt.hist(delta_E[y_true == 1], bins=40, alpha=0.7, label="Deepfake", color="#d62828")
plt.xlabel("ΔEnergy (E_after - E_before)")
plt.ylabel("Count")
plt.title("Distribution of ΔEnergy Scores (Real vs Deepfake)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_PREFIX + "hist.png", dpi=300)
plt.show()

# -------- ROC PLOT --------
fpr, tpr, _ = roc_curve(y_true, delta_E)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="#d62828", lw=2, label=f"ΔEnergy ROC (AUC={roc_auc:.4f})")
plt.plot([0,1],[0,1],'--',color='gray',lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for ΔEnergy-based Detection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_PREFIX + "roc.png", dpi=300)
plt.show()

print(f"\nSaved plots to: {SAVE_PREFIX}hist.png and {SAVE_PREFIX}roc.png")
