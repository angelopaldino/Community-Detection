"""
Visualizzazione del grafo di Cora con layout a due livelli.

Il layout a forze standard (spring_layout) su un grafo sparso di ~2700 nodi
produce un "hairball" in cui la struttura a community non e' leggibile.
Si adotta invece un layout gerarchico: le community individuate da Louvain
vengono disposte su un cerchio, e all'interno di ciascuna si applica uno
spring layout locale.

Colorando poi i nodi per CATEGORIA VERA si ottiene una verifica visiva
dell'omofilia: se i blocchi strutturali risultano cromaticamente omogenei,
struttura e tema coincidono.
"""

import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import community as community_louvain

COLORI_CAT = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
              "#f58231", "#911eb4", "#42d4f4"]
NOMI_CLASSI = ["Case_Based", "Genetic_Algorithms", "Neural_Networks",
               "Probabilistic_Methods", "Reinforcement_Learning",
               "Rule_Learning", "Theory"]


def layout_gerarchico(G, gruppi, min_dim=15, raggio=1.0, k_intra=0.45, seed=42):
    """
    Dispone i baricentri delle community su un cerchio (raggio proporzionale
    alla numerosita') e applica uno spring layout dentro ciascuna.
    Le community sotto `min_dim` nodi vengono accorpate in una regione esterna,
    per non affollare il cerchio con centinaia di micro-gruppi.
    """
    membri = {}
    for v, c in gruppi.items():
        membri.setdefault(c, []).append(v)

    grandi = sorted([c for c, m in membri.items() if len(m) >= min_dim],
                    key=lambda c: -len(membri[c]))
    piccole = [c for c in membri if c not in grandi]

    n = len(grandi)
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

    # micro-community e nodi isolati: anello esterno
    rng = np.random.default_rng(seed)
    resto = [v for c in piccole for v in membri[c]]
    for j, v in enumerate(resto):
        ang = 2 * np.pi * j / max(len(resto), 1)
        r = scala * (1.55 + 0.12 * rng.random())
        pos[v] = r * np.array([np.cos(ang), np.sin(ang)])
    return pos


def disegna(G, pos, colori, titolo, path, legenda=True, dim_nodo=11):
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.25)
    nx.draw_networkx_nodes(G, pos, node_color=colori, node_size=dim_nodo,
                           linewidths=0)
    if legenda:
        for i, nome in enumerate(NOMI_CLASSI):
            plt.scatter([], [], c=COLORI_CAT[i], label=nome, s=45)
        plt.legend(loc="upper right", fontsize=10, frameon=True,
                   framealpha=0.95, edgecolor="#cccccc")
    plt.title(titolo, fontsize=15, pad=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f">> salvato: {path}")


if __name__ == "__main__":
    # --- adatta al tuo loader ---
    import sys
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora

    OUT = os.path.join(RADICE, "experiments", "visualizzazione")
    os.makedirs(OUT, exist_ok=True)

    dati = carica_cora()
    G, etichette = dati["grafo"], dati["etichette"]
    print(f"grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

    print(">> Louvain")
    part = community_louvain.best_partition(G, random_state=42)
    print(f"   {len(set(part.values()))} community")

    print(">> layout gerarchico (qualche minuto)")
    pos = layout_gerarchico(G, part)

    # (1) colori = categoria vera, posizioni = community di Louvain.
    #     verifica visiva dell'omofilia.
    disegna(G, pos, [COLORI_CAT[etichette[n]] for n in G.nodes()],
            "Cora: community strutturali (Louvain), colori per categoria tematica",
            os.path.join(OUT, "grafo_categorie.png"))

    # (2) colori = community di Louvain: la struttura pura
    cmap = plt.get_cmap("tab20")
    disegna(G, pos, [cmap(part[n] % 20) for n in G.nodes()],
            f"Cora: {len(set(part.values()))} community individuate da Louvain",
            os.path.join(OUT, "grafo_louvain.png"), legenda=False)

    print(">> fatto")