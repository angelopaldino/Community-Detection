"""
loaders.py - Loader 


Costruzione coerente:
  - NODI/indici: la riga i di cora.content e' il nodo i (stesso ordine di TAPE)
  - ARCHI:       da cora.cites, traducendo gli id-paper originali nella
                 posizione di riga (0..2707) tramite cora.content
  - TESTO:       da TAPE (titolo T + abstract A), ordinato per id
  - ETICHETTE:   da TAPE (label 0..6), ground-truth per NMI/ARI

Restituisce: grafo networkx, lista testi, array etichette, tutti allineati per nodo.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx


def carica_cora(
    path_content=r"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.content",
    path_cites="C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.cites",
    path_tape="C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed\\combined.csv",
):
    # -----------------------------------------------------------
    # 1) ORDINE DEI NODI da cora.content
    #    La riga i-esima definisce il nodo i. Costruisco la mappa
    #    id-paper-originale -> indice-di-riga (0..2707).
    # -----------------------------------------------------------
    print(">> Leggo cora.content per fissare l'ordine dei nodi...")
    content = pd.read_csv(path_content, sep="\t", header=None)
    paper_ids = content.iloc[:, 0].astype(str).values      # id-paper originali
    n_nodi = len(paper_ids)
    id_to_index = {pid: i for i, pid in enumerate(paper_ids)}
    print(f"   Nodi: {n_nodi}")

    # -----------------------------------------------------------
    # 2) ARCHI da cora.cites, tradotti in indici 0..2707
    # -----------------------------------------------------------
    print(">> Leggo cora.cites e traduco gli archi in indici di nodo...")
    cites = pd.read_csv(path_cites, sep="\t", header=None,
                        names=["cited", "citing"], dtype=str)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodi))
    archi_scartati = 0
    for cited, citing in zip(cites["cited"], cites["citing"]):
        if cited in id_to_index and citing in id_to_index:
            G.add_edge(id_to_index[citing], id_to_index[cited])
        else:
            archi_scartati += 1
    print(f"   Archi validi: {G.number_of_edges()} (scartati: {archi_scartati})")

    # -----------------------------------------------------------
    # 3) TESTO + ETICHETTE da TAPE, ordinato per id
    # -----------------------------------------------------------
    print(">> Leggo il testo di TAPE e lo ordino per id...")
    tape = pd.read_csv(path_tape).sort_values("id").reset_index(drop=True)

    if not np.array_equal(tape["id"].values, np.arange(n_nodi)):
        raise ValueError("Gli id di TAPE non sono 0..N-1: allineamento non garantito.")

    # -----------------------------------------------------------
    # 4) CONTROLLO INCROCIATO: le classi di TAPE devono combaciare
    #    con quelle di cora.content, riga per riga. (deve dare ~100%)
    # -----------------------------------------------------------
    def norm(s):
        return str(s).strip().lower().replace(" ", "_")
    classe_content = np.array([norm(x) for x in content.iloc[:, -1].values])
    classe_tape = np.array([norm(x) for x in tape["class"].values])
    concordanza = np.mean(classe_content == classe_tape)
    print(f">> Concordanza classi content vs TAPE: {concordanza*100:.1f}%")
    if concordanza < 0.99:
        raise ValueError("Le classi non combaciano: STOP, allineamento errato.")
    print("   Allineamento confermato.")

    # -----------------------------------------------------------
    # 5) COSTRUISCO IL TESTO PER NODO (titolo + abstract)
    # -----------------------------------------------------------
    testi = []
    nodi_senza_testo = []
    for i in range(n_nodi):
        titolo = tape.loc[i, "T"]
        abstract = tape.loc[i, "A"]
        titolo = "" if pd.isna(titolo) else str(titolo).strip()
        abstract = "" if pd.isna(abstract) else str(abstract).strip()
        if titolo and abstract:
            testi.append(f"{titolo}. {abstract}")
        elif titolo:
            testi.append(titolo)
        elif abstract:
            testi.append(abstract)
        else:
            testi.append("")
            nodi_senza_testo.append(i)

    # Etichette numeriche (ground-truth)
    etichette = tape["label"].values.astype(int)

    return {
        "grafo": G,
        "testi": testi,
        "etichette": etichette,
        "n_nodi": n_nodi,
        "nodi_senza_testo": nodi_senza_testo,
    }


if __name__ == "__main__":
    dati = carica_cora()
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"Nodi:              {dati['n_nodi']}")
    print(f"Archi:             {dati['grafo'].number_of_edges()}")
    print(f"Testi caricati:    {len(dati['testi'])}")
    print(f"Nodi senza testo:  {len(dati['nodi_senza_testo'])}")
    print(f"\nEsempio - testo del nodo 0:")
    print(dati["testi"][0][:300])
    print(f"\nEtichetta nodo 0:  {dati['etichette'][0]}")
    print(f"Classi distinte:   {len(np.unique(dati['etichette']))}")
