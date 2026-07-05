import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import sys, os
RADICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(RADICE)
from src.data_loader.loaders import carica_cora

dati = carica_cora()
grafo = dati["grafo"]
etichette = dati["etichette"]


def modularity(grafo, partizione):
    comunita = {}
    for nodo, c in enumerate(partizione):
        comunita.setdefault(c, set()).add(nodo)
    return nx.algorithms.community.modularity(grafo, comunita.values())


def valuta_partizione(nome, partizione, etichette, grafo):
    n_com = len(set(partizione))
    nmi = normalized_mutual_info_score(etichette, partizione)
    ari = adjusted_rand_score(etichette, partizione)
    mod = modularity(grafo, partizione)
    print(f"{nome:20s} | community: {n_com:4d} | NMI: {nmi:.4f} | ARI: {ari:.4f} | Q: {mod:.4f}")
    return {"nome": nome, "n_community": n_com, "nmi": nmi, "ari": ari, "modularity": mod}



part_louvain = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\baseline_louvain_C0.npy")
part_perplexity = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_perplexity_finale.npy")

print("="*80)
print("CONFRONTO PARTIZIONI (vs ground-truth = categorie vere)")
print("="*80)
r_louvain = valuta_partizione("Louvain", part_louvain, etichette, grafo)
r_perplexity = valuta_partizione("Perplexity (LLM)", part_perplexity, etichette, grafo)
print("="*80)

# salvo il confronto
import json
with open("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\confronto_metriche.json", "w") as f:
    json.dump({"louvain": r_louvain, "perplexity": r_perplexity}, f, indent=2)
print("Confronto salvato in C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\confronto_metriche.json")