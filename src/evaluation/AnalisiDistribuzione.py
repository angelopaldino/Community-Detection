import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
from src.data_loader.loaders import carica_cora







"""
Grafico della distribuzione dei gradi con fit power law, usando le funzioni
native del pacchetto `powerlaw`. Si usa la CCDF (complementary cumulative
distribution function), lo standard per le code pesanti: e' meno rumorosa
della pdf e i fit vengono tracciati automaticamente a partire da x_min.
"""
import numpy as np
import networkx as nx
import powerlaw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grafico_gradi(G, path):
    gradi = np.array([d for _, d in G.degree() if d > 0])
    fit = powerlaw.Fit(gradi, discrete=True, verbose=False)
    xmin = fit.power_law.xmin
    alpha = fit.power_law.alpha

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- MODIFICA PRINCIPALE: Plottiamo TUTTI i dati empirici ---
    # Passando direttamente l'array 'gradi' a powerlaw.plot_ccdf, 
    # mostriamo l'intera distribuzione senza il cut-off di xmin.
    powerlaw.plot_ccdf(gradi, ax=ax, color="#1f4e79", linewidth=0, marker="o",
                       markersize=4.5, label="dati empirici")
    
    # I fit teorici vengono mostrati a partire da xmin (comportamento corretto)
    fit.power_law.plot_ccdf(ax=ax, color="#c0392b", linestyle="--", linewidth=2.2,
                            label=f"power law ($\\alpha$={alpha:.2f})")
    fit.lognormal.plot_ccdf(ax=ax, color="#27ae60", linestyle=":", linewidth=2.2,
                            label="log-normale")

    # Evidenzio x_min come soglia verticale
    ax.axvline(xmin, color="#999999", linewidth=1.2, alpha=0.7, linestyle="-.")
    
    # Adatto la posizione del testo per xmin in base ai nuovi assi completi
    ax.text(xmin * 1.1, ax.get_ylim()[0] * 10, f"$x_{{min}}={xmin:.0f}$",
            fontsize=9.5, color="#666666", rotation=90, va="bottom")

    ax.set_xlabel("Grado  $k$", fontsize=12, labelpad=8)
    ax.set_ylabel("$P(K \\geq k)$", fontsize=12, labelpad=8)
    ax.set_title("Distribuzione cumulata dei gradi",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10.5, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.2)
    # Confronto tra Power Law e Log-Normale
    R, p_value = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    print(f"Likelihood Ratio (R): {R:.4f}, p-value: {p_value:.4f}")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"salvato: {path}  (alpha={alpha:.3f}, xmin={xmin:.0f})")


if __name__ == "__main__":
    import pandas as pd
    BASE = r"C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data"  
    content = pd.read_csv(f"{BASE}/cora.content", sep="\t", header=None)
    pid = content.iloc[:, 0].astype(str).values
    idx = {p: i for i, p in enumerate(pid)}
    cites = pd.read_csv(f"{BASE}/cora.cites", sep="\t", header=None,
                        names=["cited", "citing"], dtype=str)
    G = nx.Graph()
    G.add_nodes_from(range(len(pid)))
    for a, b in zip(cites["cited"], cites["citing"]):
        if a in idx and b in idx:
            G.add_edge(idx[b], idx[a])
    grafico_gradi(G, "powerlaw_gradi.png")

