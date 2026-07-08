import numpy as np
import networkx as nx
import pandas as pd
import os



def _cut_e_volumi(G, partizione):
    """
    Per ogni community calcola:
      - cut: numero di archi che escono dalla community
      - vol: somma dei gradi dei nodi della community (per normalized cut)
      - interni: archi interni alla community
      - size: numero di nodi
    """
    comunita = {}
    for nodo, c in enumerate(partizione):
        comunita.setdefault(c, set()).add(nodo)

    dati = {}
    for c, membri in comunita.items():
        cut = 0
        interni = 0
        vol = 0
        for u in membri:
            vol += G.degree(u)
            for v in G.neighbors(u):
                if v in membri:
                    interni += 1   # contato due volte (u-v e v-u)
                else:
                    cut += 1
        dati[c] = {
            "cut": cut,
            "vol": vol,
            "interni": interni // 2,  # ogni arco interno contato due volte
            "size": len(membri),
        }
    return dati


def conductance_media(G, partizione):
    """
    Conductance di una community = archi uscenti / min(vol(C), vol(complemento)).
    Valori bassi = community ben separata. Restituisce la media sulle community.
    """
    dati = _cut_e_volumi(G, partizione)
    vol_totale = 2 * G.number_of_edges()
    valori = []
    for c, d in dati.items():
        vol_c = d["vol"]
        vol_comp = vol_totale - vol_c
        denom = min(vol_c, vol_comp)
        if denom > 0:
            valori.append(d["cut"] / denom)
    return float(np.mean(valori)) if valori else 0.0


def normalized_cut(G, partizione):
    """
    Normalized Cut: somma su tutte le community di cut(C)/vol(C).
    Valori bassi = partizione migliore.
    """
    dati = _cut_e_volumi(G, partizione)
    nc = 0.0
    for c, d in dati.items():
        if d["vol"] > 0:
            nc += d["cut"] / d["vol"]
    return float(nc)


def ratio_cut(G, partizione):
    """
    Ratio Cut: somma su tutte le community di cut(C)/|C|.
    Valori bassi = partizione migliore.
    """
    dati = _cut_e_volumi(G, partizione)
    rc = 0.0
    for c, d in dati.items():
        if d["size"] > 0:
            rc += d["cut"] / d["size"]
    return float(rc)


def valuta_intrinseche(nome, G, partizione):
    n_com = len(set(partizione))
    cond = conductance_media(G, partizione)
    nc = normalized_cut(G, partizione)
    rc = ratio_cut(G, partizione)
    print(f"{nome:26s} | community: {n_com:4d} | conductance: {cond:.4f} | NormCut: {nc:.4f} | RatioCut: {rc:.4f}")
    return {"nome": nome, "n_community": int(n_com),
            "conductance": cond, "normalized_cut": nc, "ratio_cut": rc}



content = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.content", sep="\t", header=None)
paper_ids = content.iloc[:, 0].astype(str).values
n_nodi = len(paper_ids)
id_to_index = {pid: i for i, pid in enumerate(paper_ids)}
cites = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.cites", sep="\t", header=None, names=["cited","citing"], dtype=str)
G = nx.Graph()
G.add_nodes_from(range(n_nodi))
for cited, citing in zip(cites["cited"], cites["citing"]):
    if cited in id_to_index and citing in id_to_index:
        G.add_edge(id_to_index[citing], id_to_index[cited])
print(f"Grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

# --- elenco delle partizioni da valutare (nome, percorso file) ---
partizioni = {
    "Louvain": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\baseline_louvain_C0.npy",
    "Perplexity (LLM)": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore.npy",
    "Aggregazione Finale": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_finale.npy",
    "Refined": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_refined.npy",
    "partizione_aggregata_passata_1": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_1.npy",
    "partizione_aggregata_passata_2": f"C:\\Users\\angel\\  OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2.npy",
    "partizione_aggregata_passata_1 (1)": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_1 (1).npy",
    "partizione_aggregata_passata_2 (1)": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2 (1).npy",
    "partizione_aggregata_passata_3": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_3.npy",
    "partizione_cd": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore2.npy",
    "partizione_senzaAgg": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore.npy",
    "partizione_refined": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_refined_finale.npy",
    "partizione_perplexity_finale": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_perplexity_finale.npy",
    "partizione_aggregata_finale": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_finale.npy",
    "partizione_aggregata_intermedia": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2.npy",
    "partizione_aggregata_senzaAgg": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore.npy",
    "partizione_aggregata_refined": f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_refined_finale.npy",
    # aggiungi qui le altre che vuoi confrontare    
}

print("="*95)
print("METRICHE INTRINSECHE (senza ground-truth)")
print("="*95)
risultati = []
for nome, path in partizioni.items():
    try:
        part = np.load(path)
        if len(part) == G.number_of_nodes():
            risultati.append(valuta_intrinseche(nome, G, part))
        else:
            print(f"{nome}: SALTATA (nodi {len(part)} != {G.number_of_nodes()})")
    except FileNotFoundError:
        print(f"{nome}: file non trovato ({path})")
print("="*95)