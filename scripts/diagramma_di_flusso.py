"""
Diagramma di flusso delle riassegnazioni operate dall'approccio ibrido.

Per ogni nodo riassegnato si determina la categoria tematica DOMINANTE della
community di provenienza e di quella di destinazione. Il diagramma mostra
quanti nodi migrano da community a dominanza X verso community a dominanza Y.

Un flusso "diagonale" (X -> X) indica che il nodo, di categoria X, viene
riportato in una community tematicamente coerente con la propria etichetta:
e' la correzione semantica che ci si attende dall'obiettivo ibrido.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

COLORI_CAT = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
              "#f58231", "#911eb4", "#42d4f4"]
NOMI = ["Case_Based", "Genetic_Alg.", "Neural_Net.", "Probabilistic",
        "Reinf._Learn.", "Rule_Learn.", "Theory"]


def dominante(partizione, etichette):
    """categoria piu' frequente in ciascuna community."""
    membri = {}
    for n, c in enumerate(partizione):
        membri.setdefault(int(c), []).append(n)
    return {c: Counter(etichette[m] for m in ms).most_common(1)[0][0]
            for c, ms in membri.items()}


def matrice_flussi(part_da, part_a, etichette):
    dom_da = dominante(part_da, etichette)
    dom_a = dominante(part_a, etichette)
    M = np.zeros((7, 7), dtype=int)
    for n in range(len(part_da)):
        if part_da[n] == part_a[n]:
            continue
        M[dom_da[int(part_da[n])], dom_a[int(part_a[n])]] += 1
    return M


def disegna_sankey(M, alpha_val, path, min_flusso=3, stat=None):
    tot_out = M.sum(axis=1)
    tot_in = M.sum(axis=0)
    n_tot = M.sum()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    H = 100.0
    gap = 3.0
    scala = (H - 6*gap) / max(tot_out.sum(), 1)

    # posizioni verticali dei blocchi (sinistra = provenienza, destra = destinazione)
    def blocchi(tot):
        y, pos = 0.0, {}
        for i in range(7):
            h = tot[i] * scala
            pos[i] = (y, h)
            y += h + gap
        return pos, y

    pos_L, altL = blocchi(tot_out)
    pos_R, altR = blocchi(tot_in)
    off = (altL - altR) / 2.0

    for i in range(7):
        y, h = pos_L[i]
        if h > 0:
            ax.add_patch(plt.Rectangle((0, y), 3, h, color=COLORI_CAT[i], ec="white", lw=0.8))
            if h > 2.5:
                ax.text(-0.7, y + h/2, f"{NOMI[i]} ({tot_out[i]})", ha="right",
                        va="center", fontsize=9)
        y, h = pos_R[i]
        if h > 0:
            ax.add_patch(plt.Rectangle((27, y+off), 3, h, color=COLORI_CAT[i], ec="white", lw=0.8))
            if h > 2.5:
                ax.text(30.7, y + off + h/2, f"{NOMI[i]} ({tot_in[i]})", ha="left",
                        va="center", fontsize=9)

    # flussi
    cursL = {i: pos_L[i][0] for i in range(7)}
    cursR = {j: pos_R[j][0] + off for j in range(7)}
    ordine = sorted([(M[i,j], i, j) for i in range(7) for j in range(7)], reverse=True)
    for v, i, j in ordine:
        if v < min_flusso:
            continue
        h = v * scala
        y0, y1 = cursL[i], cursR[j]
        cursL[i] += h; cursR[j] += h
        t = np.linspace(0, 1, 60)
        s = 3*t**2 - 2*t**3    # smoothstep
        xs = 3 + 24*t
        top = (y0 + h) + ((y1 + h) - (y0 + h))*s
        bot = y0 + (y1 - y0)*s
        diag = (i == j)
        ax.fill_between(xs, bot, top, color=COLORI_CAT[i],
                        alpha=0.55 if diag else 0.25,
                        edgecolor="#333333" if diag else "none",
                        linewidth=0.5 if diag else 0)

    diagonale = int(np.trace(M))
    ax.set_xlim(-9, 39); ax.set_ylim(-4, max(altL, altR) + 4)
    ax.axis("off")
    ax.text(1.5, max(altL, altR) + 1.5, "community di provenienza", ha="center",
            fontsize=10, style="italic", color="#555555")
    ax.text(28.5, max(altL, altR) + 1.5, "community di destinazione", ha="center",
            fontsize=10, style="italic", color="#555555")
    sub = (f"flussi diagonali: {diagonale} ({100*diagonale/max(n_tot,1):.0f}\\%)")
    if stat is not None:
        sub += (f"  \u2014  attesi per caso: {100*stat['diagonale_attesa']/max(n_tot,1):.0f}\\%"
                f"  ({stat['rapporto']:.1f}$\\times$)")
    ax.set_title(f"Riassegnazioni dell'approccio ibrido ($\\alpha$={alpha_val}): "
                 f"{n_tot} nodi\n" + sub,
                 fontsize=12.5, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f">> salvato: {path}")


def analisi_significativita(M):
    """
    Confronta i flussi diagonali osservati con quelli attesi in caso di
    indipendenza fra dominanza di provenienza e di destinazione (prodotto dei
    marginali), e testa l'ipotesi di casualita' con un chi-quadro.
    """
    from scipy.stats import chi2_contingency

    tot = int(M.sum())
    diag = int(np.trace(M))
    out = M.sum(axis=1)
    ing = M.sum(axis=0)
    atteso = np.outer(out, ing) / max(tot, 1)
    diag_att = float(np.trace(atteso))

    chi2, p, dof, _ = chi2_contingency(M)

    print("\n=== Significativita' dei flussi diagonali ===")
    print(f"osservati : {diag} su {tot} ({100*diag/max(tot,1):.1f}%)")
    print(f"attesi    : {diag_att:.1f} ({100*diag_att/max(tot,1):.1f}%) "
          f"[indipendenza]")
    print(f"rapporto  : {diag/max(diag_att,1e-9):.2f}x")
    print(f"chi2 = {chi2:.1f}, dof = {dof}, p = {p:.2e}")
    print("-> i flussi NON sono casuali" if p < 0.05
          else "-> compatibili con la casualita'")

    print("\nper categoria:")
    print(f"{'categoria':>15} {'uscite':>8} {'entrate':>9} {'diag':>7} "
          f"{'diag att.':>11} {'rapporto':>10}")
    for i in range(7):
        r = M[i, i] / atteso[i, i] if atteso[i, i] > 0 else float("nan")
        print(f"{NOMI[i]:>15} {out[i]:8} {ing[i]:9} {M[i,i]:7} "
              f"{atteso[i,i]:11.1f} {r:9.2f}x")

    return {"totale": tot, "diagonale": diag,
            "diagonale_attesa": diag_att,
            "rapporto": diag / max(diag_att, 1e-9),
            "chi2": float(chi2), "p_value": float(p)}


def stampa_matrice(M):
    print("\nMatrice dei flussi (righe = dominanza provenienza, colonne = destinazione)")
    print(" " * 15 + "".join(f"{n[:8]:>10}" for n in NOMI))
    for i in range(7):
        print(f"{NOMI[i]:>14} " + "".join(f"{M[i,j]:>10}" for j in range(7)))
    print(f"\ntotale: {M.sum()} | sulla diagonale: {np.trace(M)} "
          f"({100*np.trace(M)/max(M.sum(),1):.1f}%)")


if __name__ == "__main__":
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora

    RES = os.path.join(RADICE, "experiments", "results")
    OUT = os.path.join(RADICE, "experiments", "visualizzazione")
    os.makedirs(OUT, exist_ok=True)
    ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7

    dati = carica_cora()
    etichette = dati["etichette"]
    part_im = np.load(os.path.join(RES, "baseline_infomap.npy"))
    part_ib = np.load(os.path.join(RES, f"ibrido_alpha{ALPHA}_iter4.npy"))

    M = matrice_flussi(part_im, part_ib, etichette)
    stampa_matrice(M)
    stat = analisi_significativita(M)
    disegna_sankey(M, ALPHA, os.path.join(OUT, f"flusso_ibrido_alpha{ALPHA}.png"), stat=stat)