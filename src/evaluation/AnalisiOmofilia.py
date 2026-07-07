import numpy as np
import networkx as nx
import pandas as pd


content = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.content", sep="\t", header=None)
paper_ids = content.iloc[:, 0].astype(str).values
n_nodi = len(paper_ids)
id_to_index = {pid: i for i, pid in enumerate(paper_ids)}

cites = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.cites", sep="\t", header=None, names=["cited", "citing"], dtype=str)
G = nx.Graph()
G.add_nodes_from(range(n_nodi))
for cited, citing in zip(cites["cited"], cites["citing"]):
    if cited in id_to_index and citing in id_to_index:
        G.add_edge(id_to_index[citing], id_to_index[cited])

tape = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed\\combined.csv").sort_values("id").reset_index(drop=True)
etichette = tape["label"].values.astype(int)
print(f"Nodi: {n_nodi}, Archi: {G.number_of_edges()}, Classi: {len(set(etichette))}")



def analisi_omofilia(G, etichette):
    """
    Misura l'omofilia del grafo rispetto alle categorie vere:
    quanto spesso gli archi collegano nodi della stessa categoria.
    """
    archi_interni = 0   # archi tra nodi della stessa categoria
    archi_esterni = 0   # archi tra categorie diverse
    for u, v in G.edges():
        if etichette[u] == etichette[v]:
            archi_interni += 1
        else:
            archi_esterni += 1
    tot = archi_interni + archi_esterni
    frazione_interni = archi_interni / tot if tot else 0

    # baseline: frazione attesa se gli archi fossero casuali
    # (probabilita' che due nodi a caso abbiano la stessa categoria)
    _, conteggi = np.unique(etichette, return_counts=True)
    n = len(etichette)
    p_stessa_cat = sum((c / n) ** 2 for c in conteggi)

    print(f"Archi intra-categoria: {archi_interni} ({frazione_interni:.1%})")
    print(f"Archi inter-categoria: {archi_esterni} ({1-frazione_interni:.1%})")
    print(f"Frazione attesa se casuale: {p_stessa_cat:.1%}")
    print(f"Rapporto omofilia (osservato/atteso): {frazione_interni/p_stessa_cat:.2f}x")

    # coefficiente di assortativita' di Newman (networkx)
    nx.set_node_attributes(
        G,
        {i: int(etichette[i]) for i in range(len(etichette))},
        name="categoria"
    )
    assortativita = nx.attribute_assortativity_coefficient(G, "categoria")
    print(f"Assortativita' (Newman): {assortativita:.4f}")
    return {
        "frazione_intra": frazione_interni,
        "frazione_attesa": p_stessa_cat,
        "assortativita": assortativita,
    }

risultato = analisi_omofilia(G, etichette)
