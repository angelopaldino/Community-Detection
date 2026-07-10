import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

dati = {
    0.0: (278, 3.4477, 28.6817, 6.3837),
    0.3: (278, 3.4213, 27.4310, 6.4263),
    0.5: (276, 3.4177, 27.2733, 6.4942),
    0.7: (267, 3.4131, 17.7083, 6.6528),
    0.9: (266, 3.3925, 17.4687, 7.1088),
}
alpha = np.array(sorted(dati))
ncom  = np.array([dati[a][0] for a in alpha])
ppl   = np.array([dati[a][2] for a in alpha])
colori = {0.0:"#1f4e79", 0.3:"#f39c12", 0.5:"#e67e22", 0.7:"#c0392b", 0.9:"#27ae60"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.grid":True,"grid.color":"#e8e8e8","grid.linewidth":0.6})
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(alpha, ppl, "-", color="#c0392b", linewidth=2.2, zorder=2)
for i, a in enumerate(alpha):
    ax.scatter(a, ppl[i], s=170, c=colori[a], edgecolors="white",
               linewidths=1.6, zorder=4)
    ax.annotate(f"{ppl[i]:.1f}", (a, ppl[i]), fontsize=9, color="#c0392b",
                fontweight="bold", xytext=(0, 11), textcoords="offset points", ha="center")
ax.axvspan(0.5, 0.7, color="#c0392b", alpha=0.06, zorder=0)
ax.text(0.6, 23.0, "", fontsize=12, color="#c0392b", fontweight="bold",
        ha="center", bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                               edgecolor="#c0392b", alpha=0.95))
ax.text(0.6, 20.4, "", fontsize=8.5,
        color="#c0392b", style="italic", ha="center", alpha=0.85)

ax.set_xlabel("$\\alpha$  (peso della componente semantica)", fontsize=11, labelpad=8)
ax.set_ylabel("Perplessità  PPL(C)", color="#c0392b", fontsize=11, labelpad=8)
ax.tick_params(axis="y", labelcolor="#c0392b")
ax.set_ylim(12, 32)

axb = ax.twinx()
axb.plot(alpha, ncom, "s--", color="#1f4e79", linewidth=1.8, markersize=7,
         markeredgecolor="white", markeredgewidth=1.2, zorder=3)
axb.set_ylabel("Numero di community", color="#1f4e79", fontsize=11, labelpad=8)
axb.tick_params(axis="y", labelcolor="#1f4e79")
axb.set_ylim(260, 284); axb.grid(False)

# legenda FUORI dal grafico, in basso a sinistra
ax.legend(handles=[
    Line2D([0],[0], color="#c0392b", marker="o", linewidth=2.2, markersize=8, label="PPL(C)"),
    Line2D([0],[0], color="#1f4e79", marker="s", linestyle="--", linewidth=1.8,
           markersize=7, label="Numero di community")],
    fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
    framealpha=0.95, edgecolor="#cccccc")

ax.set_title("Perplessità e granularità della partizione al variare di $\\alpha$",
             fontsize=12.5, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig("fig3_ppl.png", dpi=200, bbox_inches="tight")
print("salvato fig3_ppl.png")
plt.close()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

dati = {
    0.0: (278, 3.4477, 28.6817, 6.3837),
    0.3: (278, 3.4213, 27.4310, 6.4263),
    0.5: (276, 3.4177, 27.2733, 6.4942),
    0.7: (267, 3.4131, 17.7083, 6.6528),
    0.9: (266, 3.3925, 17.4687, 7.1088),
}
alpha = np.array(sorted(dati))
h_sem = np.array([dati[a][1] for a in alpha])
lc    = np.array([dati[a][3] for a in alpha])
l_ibr = alpha * h_sem + (1 - alpha) * lc
colori = {0.0:"#1f4e79", 0.3:"#f39c12", 0.5:"#e67e22", 0.7:"#c0392b", 0.9:"#27ae60"}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.grid":True,"grid.color":"#e8e8e8","grid.linewidth":0.6})
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(alpha, lc, "^--", color="#1f4e79", linewidth=1.7, markersize=7,
        markeredgecolor="white", markeredgewidth=1.1, zorder=2, alpha=0.85)
ax.plot(alpha, h_sem, "v--", color="#c0392b", linewidth=1.7, markersize=7,
        markeredgecolor="white", markeredgewidth=1.1, zorder=2, alpha=0.85)
ax.plot(alpha, l_ibr, "-", color="#7d3c98", linewidth=2.4, zorder=3)
for i, a in enumerate(alpha):
    ax.scatter(a, l_ibr[i], s=170, c=colori[a], edgecolors="white",
               linewidths=1.6, zorder=5)
    ax.annotate(f"{l_ibr[i]:.2f}", (a, l_ibr[i]), fontsize=9, color="#7d3c98",
                fontweight="bold", xytext=(0, -20), textcoords="offset points", ha="center")

ax.set_xlabel("$\\alpha$  (peso della componente semantica)", fontsize=11, labelpad=8)
ax.set_ylabel("bit", fontsize=11, labelpad=8)
ax.set_ylim(3.0, 7.6)

# legenda FUORI dal grafico, in basso a sinistra
ax.legend(handles=[
    Line2D([0],[0], color="#7d3c98", marker="o", linewidth=2.4, markersize=8,
           label="$L_{ibrida}(C) = \\alpha H_{sem} + (1-\\alpha) L_{strut}$"),
    Line2D([0],[0], color="#1f4e79", marker="^", linestyle="--", linewidth=1.7,
           markersize=7, label="$L_{strut}(C)$"),
    Line2D([0],[0], color="#c0392b", marker="v", linestyle="--", linewidth=1.7,
           markersize=7, label="$H_{sem}(C)$")],
    fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3,
    framealpha=0.95, edgecolor="#cccccc")

ax.set_title("Obiettivo ibrido e sue componenti al variare di $\\alpha$",
             fontsize=12.5, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig4_obiettivo_ibrido.png", dpi=200, bbox_inches="tight")
print("salvato fig4_obiettivo_ibrido.png")
plt.close()