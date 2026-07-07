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


def analisi_centralita(G, testi=None, top_k=10):
    """
    Calcola le principali misure di centralita' sul grafo e
    riporta i nodi piu' centrali per ciascuna.
    Misure: degree, closeness, betweenness, eigenvector, Katz, PageRank.
    """
    print("Calcolo delle misure di centralita'...\n")

    # 1) Degree centrality: popolarita' locale (numero di connessioni)
    degree = nx.degree_centrality(G)

    # 2) Closeness centrality: vicinanza media a tutti gli altri nodi
    closeness = nx.closeness_centrality(G)

    # 3) Betweenness centrality: ruolo di "ponte" nei cammini minimi
    betweenness = nx.betweenness_centrality(G)

    # 4) Eigenvector centrality: importanza basata sull'importanza dei vicini
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    # 5) Katz centrality: come eigenvector ma con un termine base per ogni nodo
    #    alpha deve essere < 1/lambda_max per convergere
    try:
        katz = nx.katz_centrality_numpy(G, alpha=0.005)
    except Exception:
        katz = {n: 0.0 for n in G.nodes()}

    # 6) PageRank: importanza secondo il modello del random surfer
    pagerank = nx.pagerank(G)

    misure = {
        "degree": degree,
        "closeness": closeness,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
        "katz": katz,
        "pagerank": pagerank,
    }

    # statistiche di rete (network-level)
    print("=== Statistiche a livello di rete ===")
    print(f"Densita': {nx.density(G):.5f}")
    print(f"Grado medio: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    componenti = list(nx.connected_components(G))
    print(f"Componenti connesse: {len(componenti)}")
    print(f"Nodi nella componente gigante: {len(max(componenti, key=len))}")
    print(f"Coefficiente di clustering medio: {nx.average_clustering(G):.4f}\n")

    # top-k nodi per ciascuna misura
    for nome, valori in misure.items():
        top = sorted(valori.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"=== Top {top_k} per {nome} ===")
        for nodo, score in top:
            testo_breve = ""
            if testi is not None and testi[nodo]:
                testo_breve = " | " + testi[nodo][:60].replace("\n", " ")
            print(f"  nodo {nodo:4d}: {score:.4f}{testo_breve}")
        print()

    return misure

risultato = analisi_centralita(G, testi=tape["T"].values, top_k=10)