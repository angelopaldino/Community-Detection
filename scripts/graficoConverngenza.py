"""
Traiettorie di PPL(C) e H_sem(C) lungo le iterazioni del local-moving ibrido.
PPL (media aritmetica) oscilla; H_sem (media dei log) resta quasi piatta:
i picchi di PPL sono dovuti a pochi nodi con perplessita' estrema, non a un
peggioramento diffuso della partizione.
"""
import numpy as np
import matplotlib.pyplot as plt

ppl = {
    0.3: [28.1609, 27.4400, 27.4294, 27.4310],
    0.5: [30.5314, 29.2988, 28.9108, 27.2733],
    0.7: [26.9483, 32.0967, 17.5009, 17.7083],
    0.9: [29.9833, 17.5636, 17.5050, 17.4687],
}
h_sem = {
    0.3: [3.4268, 3.4221, 3.4220, 3.4213],
    0.5: [3.4277, 3.4186, 3.4184, 3.4177],
    0.7: [3.4126, 3.4086, 3.4143, 3.4131],
    0.9: [3.4131, 3.4001, 3.3975, 3.3925],
}
PPL_INFOMAP = 28.6817
HSEM_INFOMAP = 3.4477

colori = {0.3:"#f39c12", 0.5:"#e67e22", 0.7:"#c0392b", 0.9:"#8e44ad"}
iter_x = [0, 1, 2, 3, 4]

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.grid":True,"grid.color":"#e8e8e8","grid.linewidth":0.6})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# --- SX: PPL ---
for a, serie in ppl.items():
    y = [PPL_INFOMAP] + serie
    ax1.plot(iter_x, y, "o-", color=colori[a], linewidth=2, markersize=7,
             markeredgecolor="white", markeredgewidth=1.2, label=f"$\\alpha$={a:g}")
ax1.scatter([0], [PPL_INFOMAP], s=140, c="#1f4e79", edgecolors="white",
            linewidths=1.5, zorder=5)
ax1.annotate("", (2, 32.0967), fontsize=8.5, color="#c0392b",
             style="italic", xytext=(12, -4), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1, alpha=0.6))
ax1.set_xlabel("Iterazione", fontsize=11, labelpad=7)
ax1.set_ylabel("Perplessità  PPL(C)", fontsize=11, labelpad=7)
ax1.set_xticks(iter_x)
ax1.set_title("PPL(C) — media aritmetica", fontsize=12, fontweight="bold", pad=11)
ax1.legend(fontsize=9, loc="center right", framealpha=0.95, title="peso sem.")
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# --- DX: H_sem, stessa scala relativa per confronto onesto ---
for a, serie in h_sem.items():
    y = [HSEM_INFOMAP] + serie
    ax2.plot(iter_x, y, "o-", color=colori[a], linewidth=2, markersize=7,
             markeredgecolor="white", markeredgewidth=1.2, label=f"$\\alpha$={a:g}")
ax2.scatter([0], [HSEM_INFOMAP], s=140, c="#1f4e79", edgecolors="white",
            linewidths=1.5, zorder=5)
ax2.annotate("InfoMap", (0, HSEM_INFOMAP), fontsize=8.5, color="#1f4e79",
             xytext=(8, 6), textcoords="offset points")
ax2.set_xlabel("Iterazione", fontsize=11, labelpad=7)
ax2.set_ylabel("$H_{sem}(C)$  [bit]", fontsize=11, labelpad=7)
ax2.set_xticks(iter_x)
ax2.set_title("$H_{sem}(C)$ — media dei logaritmi", fontsize=12, fontweight="bold", pad=11)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.95, title="peso sem.")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fig_convergenza_ppl.png", dpi=200, bbox_inches="tight")
print("salvato")
plt.close()