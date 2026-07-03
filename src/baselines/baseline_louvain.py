"""
baseline_louvain.py - Baseline strutturale.

Esegue Louvain sul grafo di Cora per ottenere la partizione iniziale C0
e calcola le metriche di riferimento:
  - modularity : qualita' strutturale della partizione (non serve ground-truth)
  - NMI        : accordo con le 7 categorie vere (teoria dell'informazione)
  - ARI        : accordo con le 7 categorie vere (coppie di nodi, corretto per il caso)

Questi numeri sono il TERMINE DI PARAGONE per il metodo a perplexity.
La partizione C0 sara' anche il punto di partenza dell'ottimizzazione.
"""

import sys
import os
import numpy as np


RADICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(RADICE, "src"))
from data_loader.loaders import carica_cora

import community as community_louvain   
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx


def partizione_dict_to_array(partition, n_nodi):
    """Converte il dict {nodo: community} di Louvain in un array allineato ai nodi."""
    return np.array([partition[n] for n in range(n_nodi)])


def esegui_baseline(seed=42):
    # 1) Carico grafo + etichette (Fase 1)
    dati = carica_cora()
    G = dati["grafo"]
    etichette_vere = dati["etichette"]
    n_nodi = dati["n_nodi"]

    print("\n>> Eseguo Louvain sul grafo...")
    # random_state per riproducibilita' (Louvain ha una componente stocastica)
    partition = community_louvain.best_partition(G, random_state=seed)
    comunita_trovate = partizione_dict_to_array(partition, n_nodi)

    n_comunita = len(set(partition.values()))
    print(f"   Community trovate da Louvain: {n_comunita}")
    print(f"   (categorie vere nel ground-truth: {len(np.unique(etichette_vere))})")

    # 2) Calcolo le tre metriche
    # modularity: qualita' strutturale (usa il grafo e la partizione)
    modularita = community_louvain.modularity(partition, G)
    # NMI e ARI: confronto con il ground-truth
    nmi = normalized_mutual_info_score(etichette_vere, comunita_trovate)
    ari = adjusted_rand_score(etichette_vere, comunita_trovate)

    print("\n" + "=" * 55)
    print("METRICHE DI BASELINE (Louvain)")
    print("=" * 55)
    print(f"  Modularity : {modularita:.4f}   (qualita' strutturale)")
    print(f"  NMI        : {nmi:.4f}   (accordo col ground-truth, info)")
    print(f"  ARI        : {ari:.4f}   (accordo col ground-truth, coppie)")
    print("=" * 55)
    print("\nInterpretazione rapida:")
    print(f"  - Louvain trova {n_comunita} community contro le 7 categorie vere.")
    print("  - Modularity alta = community ben separate strutturalmente.")
    print("  - NMI/ARI dicono quanto queste community somigliano ai temi reali.")

    return {
        "partition": partition,               # dict {nodo: community} = C0
        "comunita_trovate": comunita_trovate, # array allineato ai nodi
        "modularita": modularita,
        "nmi": nmi,
        "ari": ari,
        "n_comunita": n_comunita,
    }


if __name__ == "__main__":
    ris = esegui_baseline()

    # salvo la partizione C0 per usarla come punto di partenza nelle fasi successive
    out_dir = os.path.join(RADICE, "experiments", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_louvain_C0.npy")
    np.save(out_path, ris["comunita_trovate"])
    print(f"\n>> Partizione C0 salvata in: {out_path}")