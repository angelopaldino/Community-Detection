from llama_cpp import Llama

llm = Llama(
    model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    logits_all=True,
    n_ctx=2048,
    verbose=False,
)
print("Modello caricato correttamente")