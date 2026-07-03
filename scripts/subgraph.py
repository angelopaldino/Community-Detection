import os
import pickle
import numpy as np
import networkx as nx
import sys


def estrai_sottografo(dati, categorie=(4, 5), solo_componente_maggiore=True):
    G = dati["grafo"]
    testi = dati["testi"]
    etichette = dati["etichette"]

    nodi_sel = [n for n in G.nodes() if etichette[n] in categorie]
    sub = G.subgraph(nodi_sel).copy()

    if solo_componente_maggiore:
        componenti = list(nx.connected_components(sub))
        piu_grande = max(componenti, key=len)
        sub = sub.subgraph(piu_grande).copy()

    nodi_orig = sorted(sub.nodes())
    vecchio_to_nuovo = {v: i for i, v in enumerate(nodi_orig)}

    G_nuovo = nx.Graph()
    G_nuovo.add_nodes_from(range(len(nodi_orig)))
    for u, v in sub.edges():
        G_nuovo.add_edge(vecchio_to_nuovo[u], vecchio_to_nuovo[v])

    testi_nuovi = [testi[v] for v in nodi_orig]
    etichette_nuove = np.array([etichette[v] for v in nodi_orig])

    return {
        "grafo": G_nuovo,
        "testi": testi_nuovi,
        "etichette": etichette_nuove,
        "n_nodi": len(nodi_orig),
        "nodi_originali": nodi_orig,
        "categorie": categorie,
    }


def _default_path():
    cache_dir = os.path.join("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "cora_sottografo.pkl")


def salva_sottografo(sub, path=None):
    path = path or _default_path()
    with open(path, "wb") as f:
        pickle.dump(sub, f)
    print(f"Sottografo salvato: {path}")
    return path


def carica_sottografo(path=None):
    path = path or _default_path()
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora

    dati = carica_cora()
    sub = estrai_sottografo(dati, categorie=(4, 5))
    salva_sottografo(sub)

    print(f"Nodi sottografo:   {sub['n_nodi']}")
    print(f"Archi sottografo:  {sub['grafo'].number_of_edges()}")
    print(f"Categorie incluse: {sub['categorie']}")
    valori, conteggi = np.unique(sub["etichette"], return_counts=True)
    for v, c in zip(valori, conteggi):
        print(f"   categoria {v}: {c} nodi")