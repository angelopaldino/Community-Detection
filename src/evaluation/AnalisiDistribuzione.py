import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


"""
Verifica se la distribuzione dei gradi di Cora segua una power law, secondo la
metodologia di Clauset, Shalizi & Newman (2009):
  1. stima dell'esponente alpha e della soglia x_min via massima verosimiglianza;
  2. bonta' di adattamento con test di Kolmogorov-Smirnov (p-value via bootstrap);
  3. confronto con distribuzioni alternative (log-normale, esponenziale) via
     likelihood ratio di Vuong.

Una power law NON si stabilisce dal solo grafico log-log: serve il confronto
con le alternative. Spesso, su grafi piccoli, il risultato e' inconcludente.
"""

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



def analizza_powerlaw(G, path_fig="powerlaw_gradi.png"):
    gradi = np.array([d for _, d in G.degree() if d > 0])
    print(f"nodi con grado > 0: {len(gradi)}")
    print(f"grado: min={gradi.min()}, max={gradi.max()}, medio={gradi.mean():.2f}")

    # fit power law (discreta, perche' i gradi sono interi)
    fit = powerlaw.Fit(gradi, discrete=True, verbose=False)
    print(f"\n--- Fit power law ---")
    print(f"alpha (esponente) : {fit.power_law.alpha:.3f}")
    print(f"x_min             : {fit.power_law.xmin}")
    print(f"sigma (err. alpha): {fit.power_law.sigma:.3f}")

    # confronto con alternative: R>0 favorisce la power law, R<0 l'alternativa;
    # p indica se la differenza e' significativa
    print(f"\n--- Confronto con distribuzioni alternative ---")
    for alt in ["lognormal", "exponential", "truncated_power_law"]:
        R, p = fit.distribution_compare("power_law", alt,
                                        normalized_ratio=True)
        verso = "power law" if R > 0 else alt
        signif = "significativo" if p < 0.05 else "NON significativo"
        print(f"power_law vs {alt:20}: R={R:+.3f}, p={p:.3f} "
              f"-> favorisce {verso} ({signif})")

    # figura: distribuzione empirica + fit, in scala log-log
    fig, ax = plt.subplots(figsize=(8, 6))
    fit.plot_pdf(ax=ax, color="#1f4e79", marker="o", linewidth=0,
                 markersize=5, label="dati empirici")
    fit.power_law.plot_pdf(ax=ax, color="#c0392b", linestyle="--",
                           linewidth=2, label=f"power law ($\\alpha$={fit.power_law.alpha:.2f})")
    fit.lognormal.plot_pdf(ax=ax, color="#27ae60", linestyle=":",
                           linewidth=2, label="log-normale")
    ax.set_xlabel("Grado $k$", fontsize=11)
    ax.set_ylabel("$p(k)$", fontsize=11)
    ax.set_title("Distribuzione dei gradi di Cora (scala log-log)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(path_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nfigura salvata: {path_fig}")
    return fit


if __name__ == "__main__":
    import os, sys
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora
    OUT = os.path.join(RADICE, "experiments", "visualizzazione")
    os.makedirs(OUT, exist_ok=True)
    dati = carica_cora()
    analizza_powerlaw(dati["grafo"], os.path.join(OUT, "powerlaw_gradi.png"))

