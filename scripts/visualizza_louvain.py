"""
visualizza_louvain.py - Visualizza il grafo di Cora colorato per COMMUNITY di Louvain.

Genera due immagini affiancate concettualmente:
  1. grafo_louvain.png     -> nodi colorati per community trovata da Louvain (107 gruppi)
  2. grafo_categorie.png   -> nodi colorati per categoria vera (7 classi, ground-truth)

Confrontandole si vede il divario tra struttura (Louvain) e temi reali,
cioe' la storia dietro i numeri modularity 0.81 ma NMI 0.44 / ARI 0.22.
"""

import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RADICE)

from src.data_loader.loaders import carica_cora

import community as community_louvain

OUT_DIR = os.path.join(RADICE, "experiments", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# 7 colori per le categorie vere
COLORI_CAT = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
              "#f58231", "#911eb4", "#42d4f4"]
NOMI_CLASSI = ["Case_Based", "Genetic_Algorithms", "Neural_Networks",
               "Probabilistic_Methods", "Reinforcement_Learning",
               "Rule_Learning", "Theory"]


def calcola_layout(G, seed=42):
    print(">> Calcolo il layout")
    return nx.spring_layout(G, k=0.15, iterations=700, seed=seed)


def disegna_louvain(G, pos, partition):
    n_com = len(set(partition.values()))
    cmap = cm.get_cmap("tab20", n_com)
    colori = [cmap(partition[n] % 20) for n in G.nodes()]

    plt.figure(figsize=(16, 16))
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colori, node_size=12, linewidths=0)
    plt.title(f"Cora - {n_com} community trovate da Louvain (colori = community)")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "grafo_louvain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">> Salvato: {path}")


def disegna_categorie(G, pos, etichette):
    colori = [COLORI_CAT[etichette[n]] for n in G.nodes()]
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colori, node_size=12, linewidths=0)
    for i, nome in enumerate(NOMI_CLASSI):
        plt.scatter([], [], c=COLORI_CAT[i], label=nome, s=40)
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Cora - 7 categorie vere (colori = ground-truth)")
    plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "grafo_categorie.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">> Salvato: {path}")


if __name__ == "__main__":
    dati = carica_cora()
    G = dati["grafo"]
    etichette = dati["etichette"]

    partition = community_louvain.best_partition(G, random_state=42)

    pos = calcola_layout(G)           
    disegna_louvain(G, pos, partition)
    disegna_categorie(G, pos, etichette)
    print("\n>> Fatto.")
    print("   - grafo_louvain.png  : le 107 community strutturali")
    print("   - grafo_categorie.png: i 7 temi reali")
