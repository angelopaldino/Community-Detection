import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from Spectral_Clustering7 import spectral_clustering
from Metriche import valuta_intrinseche

DATA = r"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data"
RES = r"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results"

# ricostruzione grafo ed etichette
content = pd.read_csv(f"{DATA}/cora.content", sep="\t", header=None)
paper_ids = content.iloc[:, 0].astype(str).values
n_nodi = len(paper_ids)
id_to_index = {pid: i for i, pid in enumerate(paper_ids)}
cites = pd.read_csv(f"{DATA}/cora.cites", sep="\t", header=None, names=["cited", "citing"], dtype=str)
G = nx.Graph()
G.add_nodes_from(range(n_nodi))
for cited, citing in zip(cites["cited"], cites["citing"]):
    if cited in id_to_index and citing in id_to_index:
        G.add_edge(id_to_index[citing], id_to_index[cited])
tape = pd.read_csv(f"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed\\combined.csv").sort_values("id").reset_index(drop=True)
etichette = tape["label"].values.astype(int)
print(f"Grafo completo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

# --- componente gigante ---
gigante = max(nx.connected_components(G), key=len)
nodi_gigante = sorted(gigante)
G_gig = G.subgraph(nodi_gigante).copy()
rimappa = {nodo: i for i, nodo in enumerate(nodi_gigante)}
G_rimap = nx.relabel_nodes(G_gig, rimappa)
etichette_gig = np.array([etichette[nodo] for nodo in nodi_gigante])
print(f"Componente gigante: {G_gig.number_of_nodes()} nodi, {G_gig.number_of_edges()} archi\n")

def restringi_a_gigante(partizione):
    # prendo solo i valori dei nodi della gigante, nell'ordine rimappato
    return np.array([partizione[nodo] for nodo in nodi_gigante])

# --- partizioni da valutare (nome: percorso) ---
partizioni = {
    "Louvain": f"{RES}/baseline_louvain_C0.npy",
    "Perplexity (LLM)": f"{RES}/partizione_migliore.npy",
    "Perplexity finale": f"{RES}/partizione_perplexity_finale.npy",
    "Aggregazione finale": f"{RES}/partizione_aggregata_finale.npy",
    "Refined": f"{RES}/partizione_refined_finale.npy",
}

print("="*100)
print("CONFRONTO SULLA COMPONENTE GIGANTE (tutti i metodi sullo stesso terreno)")
print("="*100)

# spectral clustering k=7 sulla gigante
print("Calcolo spectral clustering (k=7) sulla gigante...")
part_spec = spectral_clustering(G_rimap, k=7)
nmi = normalized_mutual_info_score(etichette_gig, part_spec)
ari = adjusted_rand_score(etichette_gig, part_spec)
r = valuta_intrinseche("Spectral (k=7)", G_rimap, part_spec)
print(f"  -> NMI: {nmi:.4f} | ARI: {ari:.4f}\n")

# le altre partizioni, ristrette alla gigante
for nome, path in partizioni.items():
    try:
        part = np.load(path)
        if len(part) != n_nodi:
            print(f"{nome}: SALTATA (nodi {len(part)} != {n_nodi})")
            continue
        part_gig = restringi_a_gigante(part)
        nmi = normalized_mutual_info_score(etichette_gig, part_gig)
        ari = adjusted_rand_score(etichette_gig, part_gig)
        n_com = len(set(part_gig))
        cond = valuta_intrinseche(nome, G_rimap, part_gig)
        print(f"  -> NMI: {nmi:.4f} | ARI: {ari:.4f}\n")
    except FileNotFoundError:
        print(f"{nome}: file non trovato\n")
print("="*100)