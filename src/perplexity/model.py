import numpy as np
from llama_cpp import Llama


class ModelloPerplexity:
    def __init__(self, model_path, n_ctx=4096, n_threads=None):
        self.llm = Llama(
            model_path=model_path,
            logits_all=True,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def _tokenizza(self, testo):
        # bytes -> lista di token id
        return self.llm.tokenize(testo.encode("utf-8"), add_bos=True)

    def logprob_testo(self, testo_nodo, contesto=""):
        """
        Restituisce (somma_logprob, n_token) del testo_nodo condizionato sul contesto.
        La perplexity si ricava poi come exp(-somma_logprob / n_token).
        Se contesto == "" misura la perplexity non condizionata del testo.
        """
        if contesto:
            prompt = contesto + "\n\n" + testo_nodo
            # numero di token occupati dal solo contesto (+ separatore)
            n_ctx_tok = len(self._tokenizza(contesto + "\n\n"))
        else:
            prompt = testo_nodo
            n_ctx_tok = 1  # salto solo il token BOS iniziale

        token_ids = self._tokenizza(prompt)

        # valuto il prompt per ottenere i logits token per token
        self.llm.reset()
        self.llm.eval(token_ids)

        somma_logprob = 0.0
        n_token = 0
        # per ogni posizione i, il modello predice il token i+1:
        # uso i logits alla posizione i-1 per valutare il token in posizione i
        for i in range(n_ctx_tok, len(token_ids)):
            logits = self.llm.scores[i - 1]
            logprobs = self._log_softmax(logits)
            somma_logprob += logprobs[token_ids[i]]
            n_token += 1

        return somma_logprob, n_token

    @staticmethod
    def _log_softmax(logits):
        logits = np.asarray(logits, dtype=np.float64)
        m = np.max(logits)
        logsumexp = m + np.log(np.sum(np.exp(logits - m)))
        return logits - logsumexp

    def perplexity_testo(self, testo_nodo, contesto=""):
        somma_logprob, n_token = self.logprob_testo(testo_nodo, contesto)
        if n_token == 0:
            return float("inf")
        return float(np.exp(-somma_logprob / n_token))