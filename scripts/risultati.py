import numpy as np
import matplotlib.pyplot as plt

# dati: alpha -> (n_com, h_sem, ppl, codelength, nmi, ari, modularity)
dati = {
    0.0: (278, 3.4290, 28.6817, 6.3837, 0.4136, 0.0503, 0.7309),   # InfoMap (h_sem stimato)
    0.3: (278, 3.4213, 27.4310, 6.4263, 0.4166, 0.0510, 0.7278),
    0.5: (276, 3.4177, 27.2733, 6.4942, 0.4177, 0.0518, 0.7209),
    0.9: (266, 3.3925, 17.4687, 7.1088, 0.4182, 0.0527, 0.6586),
}
alpha = np.array(sorted(dati))
h_sem = np.array([dati[a][1] for a in alpha])
lc    = np.array([dati[a][3] for a in alpha])
nmi   = np.array([dati[a][4] for a in alpha])
ari   = np.array([dati[a][5] for a in alpha])
mod   = np.array([dati[a][6] for a in alpha])

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.grid":True,"grid.color":"#e8e8e8","grid.linewidth":0.6})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

# --- SX: curva di trade-off L(C) vs H_sem ---
ax1.plot(lc, h_sem, "-", color="#bbbbbb", linewidth=1.2, zorder=1)
colori = ["#1f4e79", "#e67e22", "#c0392b", "#27ae60"]
for i, a in enumerate(alpha):
    lbl = "InfoMap ($\\alpha$=0)" if a == 0 else f"$\\alpha$={a}"
    ax1.scatter(lc[i], h_sem[i], s=150, c=colori[i], edgecolors="white",
                linewidths=1.5, zorder=3, label=lbl)
    ax1.annotate(f"{a:g}", (lc[i], h_sem[i]), fontsize=9,
                 xytext=(8, 5), textcoords="offset points")
ax1.set_xlabel("$L_{strut}(C)$  [bit]  —  compressione strutturale", labelpad=8)
ax1.set_ylabel("$H_{sem}(C)$  [bit]  —  entropia semantica", labelpad=8)
ax1.set_title("Curva di trade-off struttura–semantica", fontweight="bold", pad=12)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.annotate("", xy=(lc[-1]-0.05, h_sem[-1]+0.005), xytext=(lc[0]+0.05, h_sem[0]-0.005),
             arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2, alpha=0.7))
ax1.text(0.5*(lc[0]+lc[-1]), 0.5*(h_sem[0]+h_sem[-1])+0.008,
         "$\\alpha$ crescente", fontsize=9, color="#666666", style="italic",
         ha="center", rotation=-32)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
ax1.margins(0.14)

# --- DX: metriche in funzione di alpha (assi doppi) ---
ax2.plot(alpha, nmi, "o-", color="#c0392b", linewidth=2, markersize=8,
         markeredgecolor="white", markeredgewidth=1.2, label="NMI")
ax2.plot(alpha, ari, "s-", color="#e67e22", linewidth=2, markersize=7,
         markeredgecolor="white", markeredgewidth=1.2, label="ARI")
ax2.set_xlabel("$\\alpha$  (peso della componente semantica)", labelpad=8)
ax2.set_ylabel("NMI  /  ARI", labelpad=8)
ax2.set_ylim(0, 0.5)

ax2b = ax2.twinx()
ax2b.plot(alpha, mod, "^--", color="#1f4e79", linewidth=1.8, markersize=7,
          markeredgecolor="white", markeredgewidth=1.2, label="modularity")
ax2b.set_ylabel("modularity", color="#1f4e79", labelpad=8)
ax2b.tick_params(axis="y", labelcolor="#1f4e79")
ax2b.grid(False)

linee = ax2.get_lines() + ax2b.get_lines()
ax2.legend(linee, [l.get_label() for l in linee], fontsize=9, loc="center left")
ax2.set_title("Metriche al variare di $\\alpha$", fontweight="bold", pad=12)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("fig_ibrido_tradeoff.png", dpi=200, bbox_inches="tight")
print("salvato fig_ibrido_tradeoff.png")
plt.close()