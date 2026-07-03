"""
visualizza_grafo.py - Visualizzazione del grafo di Cora.

Con 2708 nodi il grafo intero e' un groviglio ("hairball"), quindi offriamo
due viste:
  1. INTERO   - tutti i nodi, colorati per categoria, con layout a forze.
                Utile per vedere se le categorie si raggruppano (omofilia).
  2. CAMPIONE - un sottografo piu' piccolo (es. la componente attorno ad
                alcuni nodi) per leggere davvero la struttura locale.

Richiede: networkx, matplotlib, e il loader della Fase 1.
"""

import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# importa il loader della Fase 1 (adatta il percorso al tuo progetto)
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(RADICE, "src"))

from data_loader.loaders import carica_cora
# 7 colori per le 7 categorie
COLORI = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8",
          "#f58231", "#911eb4", "#42d4f4"]
NOMI_CLASSI = ["Case_Based", "Genetic_Algorithms", "Neural_Networks",
               "Probabilistic_Methods", "Reinforcement_Learning",
               "Rule_Learning", "Theory"]


def visualizza_intero(dati, output="C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\visualizzazione\\visualizzazione_grafo_intero.png"):
    G = dati["grafo"]
    etichette = dati["etichette"]
    colori_nodi = [COLORI[etichette[n]] for n in G.nodes()]

    print(">> Calcolo il layout")
    # spring_layout: i nodi connessi si attraggono. Con k piccolo e poche
    # iterazioni resta gestibile. seed per riproducibilita'.
    pos = nx.spring_layout(G, k=0.15, iterations=30, seed=42)

    plt.figure(figsize=(16, 16))
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colori_nodi,
                           node_size=12, linewidths=0)
    # legenda
    for i, nome in enumerate(NOMI_CLASSI):
        plt.scatter([], [], c=COLORI[i], label=nome, s=40)
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Grafo di Cora - nodi colorati per categoria (ground-truth)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f">> Salvato: {output}")
    plt.close()


def visualizza_campione(dati, n_seed=3, output="C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\visualizzazione\\visualizzazione_grafo_campione.png"):
    """Sottografo leggibile: parto da alcuni nodi ad alto grado e prendo
    i loro vicini fino a 2 passi, per vedere la struttura locale."""
    G = dati["grafo"]
    etichette = dati["etichette"]

    # scelgo i nodi con piu' connessioni come semi
    gradi = dict(G.degree())
    semi = sorted(gradi, key=gradi.get, reverse=True)[:n_seed]

    nodi_sel = set(semi)
    for s in semi:
        nodi_sel.update(G.neighbors(s))
        for v in list(G.neighbors(s)):
            nodi_sel.update(G.neighbors(v))
    sub = G.subgraph(nodi_sel)
    print(f">> Sottografo campione: {sub.number_of_nodes()} nodi, {sub.number_of_edges()} archi")

    colori_nodi = [COLORI[etichette[n]] for n in sub.nodes()]
    pos = nx.spring_layout(sub, k=0.3, iterations=50, seed=42)

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_edges(sub, pos, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(sub, pos, node_color=colori_nodi,
                           node_size=60, linewidths=0.3, edgecolors="white")
    for i, nome in enumerate(NOMI_CLASSI):
        plt.scatter([], [], c=COLORI[i], label=nome, s=40)
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Cora - sottografo locale attorno ai nodi piu' connessi")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f">> Salvato: {output}")
    plt.close()


if __name__ == "__main__":
    dati = carica_cora()
    print()
    visualizza_campione(dati)   # il piu' leggibile
    visualizza_intero(dati)     # la vista d'insieme 
    print("\n>> Fatto. ")