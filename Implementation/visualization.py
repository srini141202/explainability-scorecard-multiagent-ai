import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# colours matching your HTML diagram
C_BASE1  = "#e74c3c"   # red   – Baseline 1
C_BASE2  = "#e67e22"   # orange – Baseline 2
C_YOURS  = "#27ae60"   # green  – Your system

dimensions = ["D1\nAgent\nTransparency",
              "D2\nComm.\nClarity",
              "D3\nOutcome\nAttribution",
              "D4\nHuman\nOversight",
              "D5\nTemporal\nAccountability"]

baseline1 = [1.25, 1.65, 3.40, 3.79, 0.00]
baseline2 = [7.12, 4.44, 6.70, 7.92, 5.00]
yours     = [9.62, 6.74, 10.00, 9.13, 10.00]

x     = np.arange(len(dimensions))
width = 0.26

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

b1 = ax.bar(x - width, baseline1, width, label="Baseline 1 — Single LLM call (no XAI)",
            color=C_BASE1, alpha=0.88, edgecolor="white", linewidth=0.5)
b2 = ax.bar(x,          baseline2, width, label="Baseline 2 — Basic RAG + single agent",
            color=C_BASE2, alpha=0.88, edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + width,  yours,     width, label="This system — Four-agent XAI pipeline",
            color=C_YOURS, alpha=0.88, edgecolor="white", linewidth=0.5)

# value labels on bars
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=8, color="#333333", fontweight="600")

ax.set_xticks(x)
ax.set_xticklabels(dimensions, fontsize=10, color="#333333")
ax.set_ylabel("Score (out of 10)", fontsize=11, color="#333333")
ax.set_ylim(0, 12)
ax.set_yticks(range(0, 11))
ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(colors="#555555")

ax.set_title("Explainability Scores — Baseline Comparison Across Three System Architectures",
             fontsize=12, fontweight="700", color="#1a1a1a", pad=14)

legend = ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
                   edgecolor="#cccccc", facecolor="white")

# composite score annotation
ax.annotate("Composite scores:\nBaseline 1: 2.02  |  Baseline 2: 6.24  |  This system: 9.10  (+350%)",
            xy=(0.5, 0.01), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=9,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#cccccc", linewidth=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("baseline_chart.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("baseline_chart.png saved")

# CHART 2 — Reliability CV bar chart
dims_rel = ["D1\nAgent\nTransparency",
            "D2\nComm.\nClarity",
            "D3\nOutcome\nAttribution",
            "D4\nHuman\nOversight",
            "D5\nTemporal\nAccountability",
            "Composite"]

cvs      = [0.00, 0.24, 0.00, 2.85, 0.00, 0.59]
colours  = ["#27ae60" if c < 10 else "#e74c3c" for c in cvs]

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

bars2 = ax2.bar(dims_rel, cvs, color=colours, alpha=0.88,
                edgecolor="white", linewidth=0.5, width=0.5)

for bar, cv in zip(bars2, cvs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             f"{cv:.2f}%", ha="center", va="bottom",
             fontsize=10, fontweight="700", color="#333333")

ax2.axhline(y=10, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
ax2.text(5.4, 10.15, "10% threshold", color="#e74c3c",
         fontsize=9, ha="right")

ax2.set_ylabel("Coefficient of Variation (%)", fontsize=11, color="#333333")
ax2.set_ylim(0, 13)
ax2.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#cccccc")
ax2.spines["bottom"].set_color("#cccccc")
ax2.tick_params(colors="#555555")
ax2.set_xticklabels(dims_rel, fontsize=10, color="#333333")

ax2.set_title("Reliability Test — Coefficient of Variation per Dimension (10 Repeated Runs)",
              fontsize=12, fontweight="700", color="#1a1a1a", pad=14)

green_patch = mpatches.Patch(color="#27ae60", alpha=0.88, label="RELIABLE (CV < 10%)")
ax2.legend(handles=[green_patch], loc="upper right",
           fontsize=9, framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig("reliability_chart.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("reliability_chart.png saved")