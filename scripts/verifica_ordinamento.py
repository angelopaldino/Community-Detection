"""
Verifica se l'ordinamento dei nodi di TAPE (id 0..2707) coincide con
l'ordine delle righe del file originale cora.content.

Idea: se coincidono, allora la classe testuale in content (riga i) e
la label numerica di TAPE (id i) devono avere una corrispondenza
COSTANTE (es. 'Neural_Networks' sempre = 3). Misuriamo questa coerenza.
Se ~100% -> gli ordinamenti coincidono e possiamo usare cora.cites per gli archi.
"""

import pandas as pd
import numpy as np

# --- ADATTA QUESTO PERCORSO al tuo cora.content ---
PATH_CONTENT = "C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\cora.content"
PATH_TAPE = "C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed\\combined.csv"

# 1) Leggo cora.content: prima colonna = paper_id, ultima = classe testuale
content = pd.read_csv(PATH_CONTENT, sep="\t", header=None)
paper_id_content = content.iloc[:, 0].values          # id-paper originali (sparsi)
classe_content = content.iloc[:, -1].values           # es. 'Neural_Networks'
print(f"Righe in cora.content: {len(content)}")
print(f"Esempio prime 5 classi content: {classe_content[:5]}")

# 2) Leggo TAPE, ordinato per id
tape = pd.read_csv(PATH_TAPE).sort_values("id").reset_index(drop=True)
label_tape = tape["label"].values                     # 0..6
classe_tape = tape["class"].values if "class" in tape.columns else None
print(f"\nRighe in TAPE: {len(tape)}")
print(f"Esempio prime 5 label TAPE: {label_tape[:5]}")
if classe_tape is not None:
    print(f"Esempio prime 5 class TAPE:  {classe_tape[:5]}")

# ------------------------------------------------------------------
# TEST A: se TAPE ha la colonna 'class' testuale, confronto DIRETTO
#         i nomi riga per riga (normalizzando la formattazione).
# ------------------------------------------------------------------
if classe_tape is not None and len(content) == len(tape):
    def norm(s):
        return str(s).strip().lower().replace(" ", "_")
    nomi_content = np.array([norm(x) for x in classe_content])
    nomi_tape = np.array([norm(x) for x in classe_tape])
    concordanza = np.mean(nomi_content == nomi_tape)
    print(f"\n[TEST A] Concordanza nomi classe (content vs TAPE), riga per riga: {concordanza*100:.1f}%")
    if concordanza > 0.99:
        print("  => Gli ordinamenti COINCIDONO. Possiamo usare cora.cites per gli archi.")
    else:
        print("  => Gli ordinamenti NON coincidono. Serve una mappa esplicita.")

# ------------------------------------------------------------------
# TEST B (fallback): corrispondenza nome->numero coerente.
#         Per ogni classe testuale di content, guardo quale label TAPE
#         le corrisponde piu' spesso; se la mappa e' pulita (1 a 1
#         quasi perfetta), gli ordinamenti coincidono.
# ------------------------------------------------------------------
if len(content) == len(tape):
    df = pd.DataFrame({"nome": [str(x).strip() for x in classe_content],
                       "num": label_tape})
    tabella = df.groupby("nome")["num"].agg(lambda s: s.value_counts().index[0])
    # coerenza: quante righe rispettano la mappa dominante
    mappa = tabella.to_dict()
    previsti = df["nome"].map(mappa).values
    coerenza = np.mean(previsti == df["num"].values)
    print(f"\n[TEST B] Coerenza mappa nome->numero: {coerenza*100:.1f}%")
    print("Mappa dedotta (nome classe content -> label TAPE):")
    for nome, num in tabella.items():
        print(f"   {nome:25s} -> {num}")
    if coerenza > 0.99:
        print("  => Mappa pulita: gli ordinamenti COINCIDONO.")
    else:
        print("  => Mappa sporca: gli ordinamenti sono DIVERSI.")
else:
    print("\nATTENZIONE: content e TAPE hanno numero di righe diverso, confronto non valido.")