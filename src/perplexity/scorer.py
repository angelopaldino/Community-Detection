import random
import numpy as np
from tqdm import tqdm


class ScorerPerplexity:
    def __init__(self, modello, testi, max_nodi_contesto=5, max_char_testo=600, seed=42):
        self.modello = modello
        self.testi = testi
        self.max_nodi_contesto = max_nodi_contesto
        self.max_char_testo = max_char_testo
        self.rng = random.Random(seed)

    def _costruisci_contesto(self, nodo, membri_community):
        altri = [m for m in membri_community if m != nodo]
        if not altri:
            return ""
        k = min(self.max_nodi_contesto, len(altri))
        campione = self.rng.sample(altri, k)
        pezzi = [self.testi[m][:self.max_char_testo] for m in campione if self.testi[m]]
        return "\n\n".join(pezzi)

    def perplexity_nodo(self, nodo, membri_community):
        testo = self.testi[nodo][:self.max_char_testo]
        if not testo:
            return None
        contesto = self._costruisci_contesto(nodo, membri_community)
        return self.modello.perplexity_testo(testo, contesto)

    def ppl_partizione(self, partizione, mostra_avanzamento=True, descrizione="PPL(C)"):
        community = {}
        for nodo, c in enumerate(partizione):
            community.setdefault(c, []).append(nodo)

        ppl_per_nodo = {}
        iteratore = range(len(partizione))
        if mostra_avanzamento:
            iteratore = tqdm(iteratore, desc=descrizione, unit="nodo")

        for nodo in iteratore:
            c = partizione[nodo]
            ppl = self.perplexity_nodo(nodo, community[c])
            if ppl is not None:
                ppl_per_nodo[nodo] = ppl

        if not ppl_per_nodo:
            return float("inf"), {}
        media = float(np.mean(list(ppl_per_nodo.values())))
        return media, ppl_per_nodo


if __name__ == "__main__":
    import sys, os
    RADICE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(RADICE)
    from scripts.subgraph import carica_sottografo
    from src.perplexity.model import ModelloPerplexity

    sub = carica_sottografo()
    testi = sub["testi"]
    etichette = sub["etichette"]

    MODELLO = os.path.join(RADICE, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    print("Carico il modello...")
    m = ModelloPerplexity(MODELLO)
    scorer = ScorerPerplexity(m, testi)

    print("Calcolo PPL(C) con partizione = ground-truth...")
    ppl_gt, _ = scorer.ppl_partizione(etichette, descrizione="PPL ground-truth")

    print("Calcolo PPL(C) con partizione casuale...")
    rng = np.random.default_rng(0)
    part_random = rng.integers(0, len(set(etichette)), size=len(etichette))
    ppl_rand, _ = scorer.ppl_partizione(part_random, descrizione="PPL casuale")

    print("\n" + "=" * 50)
    print(f"PPL(C) ground-truth: {ppl_gt:.3f}")
    print(f"PPL(C) casuale:      {ppl_rand:.3f}")
    print("=" * 50)
    if ppl_gt < ppl_rand:
        print("OK: la partizione vera ha perplessita' piu' bassa della casuale.")
    else:
        print("ATTENZIONE: ordine inatteso, da verificare.")