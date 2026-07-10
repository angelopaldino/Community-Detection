"""
Visualizza l'effetto dell'approccio ibrido sul grafo di Cora.

Il layout gerarchico e' costruito a partire dalla partizione di InfoMap
(il punto di partenza del local-moving ibrido): i baricentri delle community
sono disposti su un cerchio, e dentro ciascuna si applica uno spring locale.

Vengono prodotte tre figure:
  1. grafo_infomap_categorie.png  - situazione di partenza: blocchi di InfoMap,
                                    colori per categoria tematica vera.
  2. grafo_ibrido_spostati.png    - i nodi riassegnati dall'ibrido, evidenziati.
  3. grafo_ibrido_frecce.png      - i movimenti dei nodi (da/verso), con frecce.
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORI_CAT = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
              "#f58231", "#911eb4", "#42d4f4"]
NOMI_CLASSI = ["Case_Based", "Genetic_Algorithms", "Neural_Networks",
               "Probabilistic_Methods", "Reinforcement_Learning",
               "Rule_Learning", "Theory"]


def layout_gerarchico(G, gruppi, min_dim=12, k_intra=0.45, seed=42):
    """Baricentri delle community su un cerchio, spring locale dentro ciascuna."""
    membri = {}
    for v, c in gruppi.items():
        membri.setdefault(int(c), []).append(v)

    grandi = sorted([c for c, m in membri.items() if len(m) >= min_dim],
                    key=lambda c: -len(membri[c]))
    piccole = [c for c in membri if c not in grandi]

    n = max(len(grandi), 1)
    scala = 2.6 * np.sqrt(max(len(m) for m in membri.values()))
    pos = {}
    for i, c in enumerate(grandi):
        ang = 2 * np.pi * i / n
        centro = scala * np.array([np.cos(ang), np.sin(ang)])
        sub = G.subgraph(membri[c])
        p = nx.spring_layout(sub, k=k_intra, iterations=60, seed=seed)
        fatt = 0.45 * np.sqrt(len(membri[c]))
        for v, xy in p.items():
            pos[v] = centro + fatt * np.asarray(xy)

    rng = np.random.default_rng(seed)
    resto = [v for c in piccole for v in membri[c]]
    for j, v in enumerate(resto):
        ang = 2 * np.pi * j / max(len(resto), 1)
        r = scala * (1.55 + 0.12 * rng.random())
        pos[v] = r * np.array([np.cos(ang), np.sin(ang)])
    return pos


def fig_partenza(G, pos, etichette, path):
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.25)
    nx.draw_networkx_nodes(G, pos, node_size=11, linewidths=0,
                           node_color=[COLORI_CAT[etichette[n]] for n in G.nodes()])
    for i, nome in enumerate(NOMI_CLASSI):
        plt.scatter([], [], c=COLORI_CAT[i], label=nome, s=45)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#cccccc")
    plt.title("Community di InfoMap, colori per categoria tematica", fontsize=15, pad=16)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f">> salvato: {path}")


def fig_spostati(G, pos, etichette, spostati, alpha_val, path):
    fermi = [n for n in G.nodes() if n not in spostati]
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_edges(G, pos, alpha=0.035, width=0.22)
    # nodi non toccati: sbiaditi
    nx.draw_networkx_nodes(G, pos, nodelist=fermi, node_size=9, linewidths=0,
                           node_color=[COLORI_CAT[etichette[n]] for n in fermi],
                           alpha=0.28)
    # nodi riassegnati: pieni, con bordo nero
    nx.draw_networkx_nodes(G, pos, nodelist=list(spostati), node_size=34,
                           node_color=[COLORI_CAT[etichette[n]] for n in spostati],
                           edgecolors="#111111", linewidths=0.7)
    for i, nome in enumerate(NOMI_CLASSI):
        plt.scatter([], [], c=COLORI_CAT[i], label=nome, s=45)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="#cccccc")
    plt.title(f"Nodi riassegnati dall'approccio ibrido ($\\alpha$={alpha_val}): "
              f"{len(spostati)} su {G.number_of_nodes()}", fontsize=15, pad=16)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f">> salvato: {path}")


def fig_frecce(G, pos, etichette, spostati, part_ib, alpha_val, path, max_frecce=400):
    """Freccia dal nodo al baricentro della community di destinazione."""
    centri = {}
    for n in G.nodes():
        centri.setdefault(int(part_ib[n]), []).append(pos[n])
    centri = {c: np.mean(np.array(p), axis=0) for c, p in centri.items()}

    plt.figure(figsize=(15, 15))
    nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.2)
    nx.draw_networkx_nodes(G, pos, node_size=8, linewidths=0, alpha=0.22,
                           node_color=[COLORI_CAT[etichette[n]] for n in G.nodes()])
    scelti = list(spostati)[:max_frecce]
    for n in scelti:
        p0 = pos[n]; p1 = centri[int(part_ib[n])]
        plt.annotate("", xy=p1, xytext=p0,
                     arrowprops=dict(arrowstyle="->", color="#c0392b",
                                     lw=0.55, alpha=0.42,
                                     connectionstyle="arc3,rad=0.12"))
    nx.draw_networkx_nodes(G, pos, nodelist=scelti, node_size=22,
                           node_color=[COLORI_CAT[etichette[n]] for n in scelti],
                           edgecolors="#111111", linewidths=0.5)
    plt.title(f"Migrazione dei nodi verso le nuove community ($\\alpha$={alpha_val})",
              fontsize=15, pad=16)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f">> salvato: {path}")


if __name__ == "__main__":
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora

    RES = os.path.join(RADICE, "experiments", "results")
    OUT = os.path.join(RADICE, "experiments", "visualizzazione")
    os.makedirs(OUT, exist_ok=True)
    ALPHA = 0.7

    dati = carica_cora()
    G, etichette = dati["grafo"], dati["etichette"]

    part_im = np.load(os.path.join(RES, "baseline_infomap.npy"))
    part_ib = np.load(os.path.join(RES, f"ibrido_alpha{ALPHA}_iter4.npy"))
    assert len(part_im) == len(part_ib) == G.number_of_nodes()

    spostati = {n for n in G.nodes() if part_im[n] != part_ib[n]}
    print(f"grafo: {G.number_of_nodes()} nodi | InfoMap: {len(set(part_im))} community "
          f"| ibrido: {len(set(part_ib))} community | riassegnati: {len(spostati)}")

    print(">> layout gerarchico da InfoMap")
    pos = layout_gerarchico(G, {n: part_im[n] for n in G.nodes()})

    fig_partenza(G, pos, etichette, os.path.join(OUT, "grafo_infomap_categorie.png"))
    fig_spostati(G, pos, etichette, spostati, ALPHA,
                 os.path.join(OUT, "grafo_ibrido_spostati.png"))
    fig_frecce(G, pos, etichette, spostati, part_ib, ALPHA,
               os.path.join(OUT, "grafo_ibrido_frecce.png"))
    print(">> fatto")