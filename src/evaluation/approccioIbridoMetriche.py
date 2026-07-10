r"""
Decompone la description length L(C) nei suoi due termini per tutte le
partizioni dell'esperimento ibrido:

    L(C) = q*H(Q)  +  sum_i p_i*H(P_i)
           \_____/     \______________/
            inter            intra

Il primo termine e' il costo di codificare i salti FRA community, il secondo
quello dei movimenti DENTRO le community.

"""

import os
import sys
import glob
import numpy as np
import infomap


class ValutatoreDecomposto:
    def __init__(self, grafo):
        self.archi = [(int(u), int(v)) for u, v in grafo.edges()]
        self.nodi = sorted(int(n) for n in grafo.nodes())
        self._im = infomap.Infomap(silent=True, num_trials=1)
        self._im.add_links(self.archi)

    def decomponi(self, partizione):
        self._im.initial_partition = {n: int(partizione[n]) + 1 for n in self.nodi}
        self._im.run(no_infomap=True)
        return {
            "totale": float(self._im.codelength),
            "inter": float(self._im.index_codelength),
            "intra": float(self._im.module_codelength),
        }


def riga_latex(nome, n_com, d):
    return (f"{nome} & {n_com} & {d['totale']:.4f} & "
            f"{d['inter']:.3f} & {d['intra']:.3f} \\\\")


if __name__ == "__main__":
    RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(RADICE)
    from src.data_loader.loaders import carica_cora

    RES = os.path.join(RADICE, "experiments", "results")
    dati = carica_cora()
    G = dati["grafo"]
    val = ValutatoreDecomposto(G)

    # baseline InfoMap (= alpha 0) + le quattro partizioni ibride finali (iter4)
    da_valutare = [("InfoMap ($\\alpha=0$)", "baseline_infomap.npy")]
    for a in [0.3, 0.5, 0.7, 0.9]:
        da_valutare.append((f"Ibrido $\\alpha={a}$", f"ibrido_alpha{a}_iter4.npy"))

    print(f"{'partizione':<24} {'#com':>6} {'L(C)':>9} {'inter':>8} {'intra':>8}")
    print("-" * 60)
    righe = []
    for nome, file in da_valutare:
        path = os.path.join(RES, file)
        if not os.path.exists(path):
            print(f"{nome:<24}  file mancante: {file}")
            continue
        part = np.load(path)
        if len(part) != G.number_of_nodes():
            print(f"{nome:<24}  dimensione errata")
            continue
        d = val.decomponi(part)
        n_com = len(set(part))
        etichetta = nome.replace("$\\alpha=0$", "(a=0)").replace("$\\alpha=", "a=").replace("$", "")
        print(f"{etichetta:<24} {n_com:>6} {d['totale']:>9.4f} "
              f"{d['inter']:>8.3f} {d['intra']:>8.3f}")
        righe.append(riga_latex(nome, n_com, d))

    print("\n\n--- ---\n")
    for r in righe:
        print(r)