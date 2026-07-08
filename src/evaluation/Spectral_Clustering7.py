import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from Metriche import valuta_intrinseche



# ^ metti qui la cartella che contiene cora.content, cora.cites, combined.csv
 
# =========================================================
# 2) ricostruzione grafo ed etichette
# =========================================================
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
print(f"Grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi, {len(set(etichette))} classi")


def spectral_clustering(G, k=7, seed=42):
    """
    Spectral clustering basato sul Laplaciano normalizzato (normalized cut).
    Calcola i k autovettori con autovalori piu' piccoli e applica k-means.
    Restituisce una partizione (array di lunghezza n_nodi) con k community.
    """
    n = G.number_of_nodes()
    nodi = sorted(G.nodes())

    # matrice di adiacenza
    A = nx.to_scipy_sparse_array(G, nodelist=nodi, dtype=float)

    # Laplaciano normalizzato simmetrico: L_sym = I - D^-1/2 A D^-1/2
    import scipy.sparse as sp
    gradi = np.array(A.sum(axis=1)).flatten()
    gradi[gradi == 0] = 1e-10   # evito divisione per zero su nodi isolati
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(gradi))
    L_sym = sp.identity(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # i k autovettori con autovalori piu' piccoli
    from scipy.sparse.linalg import eigsh
    # eigsh con which='SM' puo' essere instabile: uso sigma=0 (shift-invert)
    try:
        autovalori, autovettori = eigsh(L_sym, k=k, sigma=0, which='LM')
    except Exception:
        # fallback: converto in denso (ok per 2708 nodi)
        L_dense = L_sym.toarray()
        vals, vecs = np.linalg.eigh(L_dense)
        autovettori = vecs[:, :k]

    # normalizzo le righe (passo standard dello spectral clustering di Ng-Jordan-Weiss)
    righe_norm = np.linalg.norm(autovettori, axis=1, keepdims=True)
    righe_norm[righe_norm == 0] = 1e-10
    U = autovettori / righe_norm

    # k-means sugli autovettori
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    etichette_cluster = km.fit_predict(U)

    # rimappo sull'ordine originale dei nodi (0..n-1)
    partizione = np.zeros(n, dtype=int)
    for idx, nodo in enumerate(nodi):
        partizione[nodo] = etichette_cluster[idx]

    componenti = list(nx.connected_components(G))
    print(f"Componenti connesse nel grafo: {len(componenti)}")
    dimensioni = sorted([len(c) for c in componenti], reverse=True)
    print(f"Dimensioni delle prime componenti: {dimensioni[:10]}")
    return partizione

print("\nCalcolo spectral clustering (k=7)...")
part_spectral = spectral_clustering(G, k=7)
np.save("spectral_k7.npy", part_spectral)
print(f"Community trovate: {len(set(part_spectral))}")
 
# =========================================================
# 4) valutazione
# =========================================================
print("\n=== Rispetto al ground-truth ===")
nmi = normalized_mutual_info_score(etichette, part_spectral)
ari = adjusted_rand_score(etichette, part_spectral)
print(f"Spectral (k=7) | NMI: {nmi:.4f} | ARI: {ari:.4f}")
 
print("\n=== Metriche intrinseche ===")
valuta_intrinseche("Spectral (k=7)", G, part_spectral)
 
# distribuzione delle dimensioni delle community (per capire se e' sbilanciato)
_, conteggi = np.unique(part_spectral, return_counts=True)
print(f"\nDimensioni community: min={conteggi.min()}, max={conteggi.max()}, "
      f"media={conteggi.mean():.1f}")
print(f"Conteggi per community: {sorted(conteggi, reverse=True)}")


print("\n=== Confronto con la componente maggiore ===")
componenti = list(nx.connected_components(G))
gigante = max(componenti, key=len)
nodi_gigante = sorted(gigante)
G_gigante = G.subgraph(nodi_gigante).copy()
print(f"Componente gigante: {G_gigante.number_of_nodes()} nodi, {G_gigante.number_of_edges()} archi")
 
# rietichetto i nodi della gigante in 0..m-1 per lo spectral
rimappa = {nodo: i for i, nodo in enumerate(nodi_gigante)}
G_rimappato = nx.relabel_nodes(G_gigante, rimappa)
etichette_gigante = np.array([etichette[nodo] for nodo in nodi_gigante])
 
# =========================================================
# spectral clustering k=7 SOLO sulla componente gigante
# =========================================================
print("\nSpectral clustering (k=7) sulla componente gigante...")
part_spectral_g = spectral_clustering(G_rimappato, k=7)
print(f"Community trovate: {len(set(part_spectral_g))}")
 
# valutazione
print("\n=== Rispetto al ground-truth (solo gigante) ===")
nmi = normalized_mutual_info_score(etichette_gigante, part_spectral_g)
ari = adjusted_rand_score(etichette_gigante, part_spectral_g)
print(f"Spectral gigante (k=7) | NMI: {nmi:.4f} | ARI: {ari:.4f}")
 
print("\n=== Metriche intrinseche (solo gigante) ===")
valuta_intrinseche("Spectral gigante (k=7)", G_rimappato, part_spectral_g)
 
_, conteggi = np.unique(part_spectral_g, return_counts=True)
print(f"\nDimensioni community: {sorted(conteggi, reverse=True)}")
 
# salvo la partizione rimappata sull'indice originale (nodi fuori dalla gigante = -1)
part_completa = np.full(n_nodi, -1, dtype=int)
for i, nodo in enumerate(nodi_gigante):
    part_completa[nodo] = part_spectral_g[i]
np.save("spectral_gigante_k7.npy", part_completa)
print("\nPartizione salvata in spectral_gigante_k7.npy (nodi fuori dalla gigante = -1)")