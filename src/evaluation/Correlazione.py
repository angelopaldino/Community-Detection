import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =========================================================
# carico i due file di risultati (adatta i percorsi)
# =========================================================
with open("ppl_partizioni.json") as f:
    ppl_data = json.load(f)
with open("codelength_partizioni.json") as f:
    lc_data = json.load(f)

# accoppio le partizioni presenti in ENTRAMBI i file
nomi = [n for n in ppl_data if n in lc_data]
if len(nomi) < 3:
    print(f"ATTENZIONE: solo {len(nomi)} partizioni in comune. "
          "La correlazione ha poco senso con pochi punti.")

ppl = np.array([ppl_data[n]["ppl_media"] for n in nomi])
lc = np.array([lc_data[n]["codelength"] for n in nomi])
n_com = np.array([ppl_data[n]["n_community"] for n in nomi])

# =========================================================
# correlazioni
# =========================================================
r_pearson, p_pearson = pearsonr(lc, ppl)
r_spearman, p_spearman = spearmanr(lc, ppl)

print("="*70)
print("CORRELAZIONE tra Description Length L(C) e Perplessita' PPL(C)")
print("="*70)
print(f"{'Partizione':24s} | {'L(C)':>10s} | {'PPL(C)':>10s} | {'#com':>5s}")
print("-"*70)
for i, nome in enumerate(nomi):
    print(f"{nome:24s} | {lc[i]:10.4f} | {ppl[i]:10.4f} | {n_com[i]:5d}")
print("-"*70)
print(f"Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.4f})")
print(f"Spearman r = {r_spearman:.4f}  (p = {p_spearman:.4f})")
print("="*70)

interpretazione = (
    "correlazione positiva: L(C) e PPL(C) crescono insieme -> il ponte teorico regge"
    if r_pearson > 0.3 else
    "correlazione debole/assente: struttura e semantica catturano cose diverse"
)
print(f"\nInterpretazione: {interpretazione}")

# =========================================================
# scatter plot
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(lc, ppl, s=80, c="steelblue", edgecolors="black", zorder=3)

# etichetto ogni punto col nome della partizione
for i, nome in enumerate(nomi):
    ax.annotate(nome, (lc[i], ppl[i]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")

# retta di tendenza
if len(nomi) >= 2:
    coef = np.polyfit(lc, ppl, 1)
    xline = np.linspace(lc.min(), lc.max(), 100)
    ax.plot(xline, coef[0]*xline + coef[1], "--", color="gray", alpha=0.7,
            label=f"tendenza (Pearson r={r_pearson:.2f})")
    ax.legend()

ax.set_xlabel("Description Length L(C)  [bit]  — compressione strutturale")
ax.set_ylabel("Perplessita' PPL(C)  — compressione semantica")
ax.set_title("Struttura vs Semantica: L(C) e PPL(C) sono correlate?")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("correlazione_lc_ppl.png", dpi=150, bbox_inches="tight")
print("\nGrafico salvato in correlazione_lc_ppl.png")
plt.show()