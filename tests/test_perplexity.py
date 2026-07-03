import sys, os
RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RADICE)
from src.perplexity.model import ModelloPerplexity

MODELLO = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
m = ModelloPerplexity(MODELLO)

# testo coerente col contesto -> perplexity bassa attesa
contesto = "Reinforcement learning is a branch of machine learning where agents learn by trial and error."
testo_coerente = "The agent maximizes a reward signal through interaction with the environment."
testo_incoerente = "The recipe requires two cups of flour and a pinch of salt."

ppl_coerente = m.perplexity_testo(testo_coerente, contesto)
ppl_incoerente = m.perplexity_testo(testo_incoerente, contesto)

print(f"PPL testo coerente:   {ppl_coerente:.2f}")
print(f"PPL testo incoerente: {ppl_incoerente:.2f}")