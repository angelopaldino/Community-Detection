"""
baseline_infomap.py - Seconda baseline: InfoMap (Fase 1).

Usa l'eseguibile standalone Infomap.exe.
"""

import sys
import os
import subprocess
import numpy as np

RADICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(RADICE)
from src.data_loader.loaders import carica_cora

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# percorso dell'eseguibile (adatta se l'hai messo altrove)
INFOMAP_EXE = os.path.join(RADICE, "bin", "Infomap.exe")
OUT_DIR = os.path.join(RADICE, "experiments", "results")
os.makedirs(OUT_DIR, exist_ok=True)


def scrivi_pajek(G, path):
    """Scrive il grafo in formato Pajek (.net), che Infomap legge nativamente.
    I nodi Pajek sono numerati da 1 (non da 0), quindi sommo 1 agli indici."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"*Vertices {G.number_of_nodes()}\n")
        for n in range(G.number_of_nodes()):
            f.write(f'{n+1} "{n}"\n')
        f.write("*Edges\n")
        for u, v in G.edges():
            f.write(f"{u+1} {v+1}\n")


def leggi_tree(path, n_nodi):
    """Legge il file .tree di Infomap e ricostruisce la partizione.
    Ogni riga: 'percorso_modulo flow nome_nodo id_nodo'
    Il primo numero del percorso (es. '1:3' -> 1) e' il modulo di primo livello."""
    comunita = np.full(n_nodi, -1, dtype=int)
    with open(path, "r", encoding="utf-8") as f:
        for riga in f:
            if riga.startswith("#") or not riga.strip():
                continue
            parti = riga.split()
            percorso = parti[0]                 # es. "1:3"
            modulo_top = int(percorso.split(":")[0])
            # il nome del nodo (quarto campo) e' il nostro indice originale
            # ma per robustezza usiamo l'id fisico se presente
            nome_nodo = parti[2].strip('"')
            idx = int(nome_nodo)
            comunita[idx] = modulo_top
    return comunita


def esegui_infomap(seed=42):
    if not os.path.exists(INFOMAP_EXE):
        raise FileNotFoundError(f"Infomap.exe non trovato in {INFOMAP_EXE}")
 
    dati = carica_cora()
    G = dati["grafo"]
    etichette_vere = dati["etichette"]
    n_nodi = dati["n_nodi"]
 
    # scrivo il grafo nella cartella di output
    net_path = os.path.join(OUT_DIR, "cora.net")
    scrivi_pajek(G, net_path)
    print(f"Grafo scritto: {net_path}")
 
    # eseguo Infomap con cwd=OUT_DIR e nomi file relativi,
    # cosi' gli spazi nel percorso assoluto non rompono gli argomenti
    comando = [INFOMAP_EXE, "cora.net", ".", "--two-level", "--seed", str(seed)]
    risultato = subprocess.run(
        comando, cwd=OUT_DIR, capture_output=True, text=True
    )
    print(risultato.stdout[-1500:])
    if risultato.returncode != 0:
        print("STDERR:", risultato.stderr)
        raise RuntimeError("Infomap ha restituito un errore.")
 
    tree_path = os.path.join(OUT_DIR, "cora.tree")
    comunita = leggi_tree(tree_path, n_nodi)
    n_comunita = len(set(comunita[comunita >= 0]))
 
    nmi = normalized_mutual_info_score(etichette_vere, comunita)
    ari = adjusted_rand_score(etichette_vere, comunita)
 
    print("\n" + "=" * 55)
    print("METRICHE DI BASELINE (InfoMap)")
    print("=" * 55)
    print(f"  Community trovate : {n_comunita}")
    print(f"  NMI               : {nmi:.4f}")
    print(f"  ARI               : {ari:.4f}")
    print("=" * 55)
 
    np.save(os.path.join(OUT_DIR, "baseline_infomap.npy"), comunita)
    return {"comunita": comunita, "nmi": nmi, "ari": ari, "n_comunita": n_comunita}


if __name__ == "__main__":
    esegui_infomap()