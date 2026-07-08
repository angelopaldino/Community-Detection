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


part_infomap = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\baseline_infomap.npy")
part_louvain = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\baseline_louvain_C0.npy")
part_perplexity = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_perplexity_finale.npy")
part_aggregazione_Finale = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_finale.npy")
part_aggregazione_intermedia = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2.npy")
partizione_senzaAgg = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore.npy")
partizione_refined = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_refined_finale.npy")
partizione_cd = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_migliore2.npy")
partizione1 = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_1.npy")
partizione2 = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2.npy")
partizione1_1 = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_1 (1).npy")
partizione2_1 = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_2 (1).npy")
partizione3 = np.load("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\partizione_aggregata_passata_3.npy")




'''
experiments/results/partizione_aggregata_passata_1.npy
 experiments/results/partizione_aggregata_passata_2.npy
   experiments/results/partizione_aggregata_passata_1 (1).npy
     experiments/results/partizione_aggregata_passata_2 (1).npy
       experiments/results/partizione_aggregata_passata_3.npy

'''



print("="*80)
print("CONFRONTO PARTIZIONI (vs ground-truth = categorie vere)")
print("="*80)
part_infomap = valuta_partizione("Infomap", part_infomap, etichette, grafo)
r_louvain = valuta_partizione("Louvain", part_louvain, etichette, grafo)
r_perplexity = valuta_partizione("Perplexity (LLM)", part_perplexity, etichette, grafo)
r_aggregazione_Finale = valuta_partizione("Aggregazione Finale", part_aggregazione_Finale, etichette, grafo)
r_aggregazione_intermedia = valuta_partizione("Aggregazione Intermedia", part_aggregazione_intermedia, etichette, grafo)
r_aggregazione_senzaAgg = valuta_partizione("Senza Agg", partizione_senzaAgg, etichette, grafo)
r_aggregazione_refined = valuta_partizione("Partizione refined", partizione_refined,etichette,grafo)
r_part2c = valuta_partizione("Partizione CD", partizione_cd, etichette, grafo)
r_part1_1 = valuta_partizione("Partizione 1 (1)", partizione1_1, etichette, grafo)
r_part2_1 = valuta_partizione("Partizione 2 (1)", partizione2_1, etichette, grafo)
r_part3 = valuta_partizione("Partizione 3", partizione3, etichette, grafo)
r_part1 = valuta_partizione("Partizione 1", partizione1, etichette, grafo)
r_part2 = valuta_partizione("Partizione 2", partizione2, etichette, grafo)

print("="*80)

# salvo il confronto
import json
with open("C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\confronto_metriche.json", "w") as f:
    json.dump({"louvain": r_louvain, "perplexity": r_perplexity, "aggregazione_finale": r_aggregazione_Finale, "aggregazione_intermedia": r_aggregazione_intermedia, "senzaagg": r_aggregazione_senzaAgg, "refined": r_aggregazione_refined, "part2c": r_part2c, "part1_1": r_part1_1, "part2_1": r_part2_1, "part3": r_part3,"part1": r_part1,"part2": r_part2}, f, indent=2)
print("Confronto salvato in C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\experiments\\results\\confronto_metriche.json")