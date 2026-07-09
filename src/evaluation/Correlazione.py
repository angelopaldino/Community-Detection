import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =========================================================
# PERCORSI
# =========================================================
PPL_PATH = r"C:\Users\angel\OneDrive\Desktop\Community Detection LLM\experiments\results\ppl_partizioni.json"
LC_PATH  = r"C:\Users\angel\OneDrive\Desktop\Community Detection LLM\experiments\results\codelength_partizioni.json"

with open(PPL_PATH) as f:
    ppl_data = json.load(f)
with open(LC_PATH) as f:
    lc_data = json.load(f)

# nomi grezzi -> (etichetta, categoria per il colore)
# Louvain e InfoMap restano distinti; gli altri diventano Exp1, Exp2, ...
MAPPA = {
    "Louvain":                ("Louvain",  "baseline"),
    "InfoMap":                ("InfoMap",  "baseline"),
    "Spectral (k=7)":         ("Exp1",     "baseline"),
    "Perplexity (LLM)":       ("Exp2",     "metodo"),
    "Perplexity finale":      ("Exp3",     "metodo"),
    "Perplexity migliore2":   ("Exp4",     "metodo"),
    "Aggregazione finale":    ("Exp5",     "aggregazione"),
    "Aggregazione passata_1": ("Exp6",     "aggregazione"),
    "Aggregazione passata_2": ("Exp7",     "aggregazione"),
    "Aggregazione passata_3": ("Exp8",     "aggregazione"),
    "Refined":                ("Exp9",     "refinement"),
}
COLORI = {"baseline":"#1f4e79","metodo":"#c0392b","aggregazione":"#e67e22","refinement":"#27ae60"}
ETICHETTA_CAT = {"baseline":"Baseline (struttura)","metodo":"Metodo proposto",
                 "aggregazione":"Fase di aggregazione","refinement":"Fase di refinement"}

nomi = [n for n in ppl_data if n in lc_data and n in MAPPA]
ppl  = np.array([ppl_data[n]["ppl_media"] for n in nomi])
lc   = np.array([lc_data[n]["codelength"] for n in nomi])
ncom = np.array([ppl_data[n]["n_community"] for n in nomi])
label = [MAPPA[n][0] for n in nomi]
cat   = [MAPPA[n][1] for n in nomi]

r_pearson, p_pearson = pearsonr(lc, ppl)
r_spearman, p_spearman = spearmanr(lc, ppl)

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,
    "axes.edgecolor":"#333333","axes.linewidth":0.8,
    "axes.grid":True,"grid.color":"#e5e5e5","grid.linewidth":0.6,
})

# =========================================================
# FIGURA 1 — L(C) vs PPL(C)  (con retta di tendenza)
# =========================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5))
coef = np.polyfit(lc, ppl, 1)
xline = np.linspace(lc.min()-0.3, lc.max()+0.3, 100)
ax.plot(xline, coef[0]*xline+coef[1], "--", color="#999999",
        linewidth=1.5, alpha=0.8, zorder=1,
        label=f"tendenza lineare (r = {r_pearson:.2f})")

viste=set()
for i,n in enumerate(nomi):
    c=cat[i]; mostra=ETICHETTA_CAT[c] if c not in viste else None; viste.add(c)
    ax.scatter(lc[i], ppl[i], s=150, c=COLORI[c], edgecolors="white",
               linewidths=1.5, zorder=3, label=mostra)
for i,n in enumerate(nomi):
    ax.annotate(label[i], (lc[i], ppl[i]), fontsize=9.5, color="#222222",
                fontweight="medium",
                xytext=(8,6), textcoords="offset points", zorder=4)

ax.set_xlabel("Description Length  L(C)  [bit]  —  minore = miglior compressione strutturale",
              fontsize=11, labelpad=8)
ax.set_ylabel("Perplessità  PPL(C)  —  minore = miglior coerenza semantica",
              fontsize=11, labelpad=8)
ax.set_title("Relazione tra compressione strutturale e coerenza semantica",
             fontsize=13.5, fontweight="bold", pad=14)
ax.text(0.03, 0.97,
        f"Pearson  r = {r_pearson:.2f}  (p = {p_pearson:.2f})\n"
        f"Spearman ρ = {r_spearman:.2f}  (p = {p_spearman:.2f})",
        transform=ax.transAxes, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"))
ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5, edgecolor="#cccccc")
ax.margins(0.12)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig1_struttura_vs_semantica.png", dpi=200, bbox_inches="tight")
print("Salvato: fig1_struttura_vs_semantica.png")

# =========================================================
# FIGURA 2 — PPL(C) vs numero di community  (senza linee)
# =========================================================
fig2, ax2 = plt.subplots(figsize=(9.5, 6.5))
viste=set()
for i,n in enumerate(nomi):
    c=cat[i]; mostra=ETICHETTA_CAT[c] if c not in viste else None; viste.add(c)
    ax2.scatter(ncom[i], ppl[i], s=150, c=COLORI[c], edgecolors="white",
                linewidths=1.5, zorder=3, label=mostra)
for i,n in enumerate(nomi):
    dx = -10 if ncom[i] > 820 else 8
    ha = "right" if ncom[i] > 820 else "left"
    ax2.annotate(label[i], (ncom[i], ppl[i]), fontsize=9.5, color="#222222",
                 fontweight="medium",
                 xytext=(dx,6), textcoords="offset points", ha=ha, zorder=4)

r_nc, p_nc = pearsonr(ncom, ppl)
ax2.set_xlabel("Numero di community", fontsize=11, labelpad=8)
ax2.set_ylabel("Perplessità  PPL(C)", fontsize=11, labelpad=8)
ax2.set_title("La perplessità decresce col numero di community",
              fontsize=13.5, fontweight="bold", pad=14)
ax2.text(0.97, 0.03, f"Pearson r = {r_nc:.2f}", transform=ax2.transAxes,
         fontsize=10, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"))
ax2.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5, edgecolor="#cccccc")
ax2.margins(0.14)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig2_ppl_vs_numcommunity.png", dpi=200, bbox_inches="tight")
print("Salvato: fig2_ppl_vs_numcommunity.png")

# tabella di corrispondenza Exp -> descrizione, utile per la didascalia
print("\nLegenda esperimenti:")
for n in nomi:
    print(f"  {MAPPA[n][0]:8s} = {n}")
print(f"\nPearson r={r_pearson:.4f} (p={p_pearson:.4f}) | Spearman r={r_spearman:.4f}")
print(f"PPL vs #community: r={r_nc:.4f}")
plt.close("all")