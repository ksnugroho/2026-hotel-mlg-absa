import os
import numpy as np
import torch
import torch.nn as nn


class FastTextEmbedding:
    """Load fastText .bin model and build an embedding matrix for a vocabulary."""

    def __init__(self, model_path: str):
        import fasttext
        print(f"Loading fastText model from {model_path} ...")
        self.ft = fasttext.load_model(model_path)
        self.dim = self.ft.get_dimension()

    def get_vector(self, word: str) -> np.ndarray:
        return self.ft.get_word_vector(word)

    def build_matrix(self, vocab: dict[str, int]) -> np.ndarray:
        """Build embedding matrix aligned to vocab indices."""
        matrix = np.zeros((len(vocab), self.dim), dtype=np.float32)
        for word, idx in vocab.items():
            matrix[idx] = self.get_vector(word)
        return matrix


def build_vocab(token_lists: list[list[str]],
                min_freq: int = 1,
                special_tokens: list[str] = None) -> dict[str, int]:
    special_tokens = special_tokens or ['<PAD>', '<UNK>']
    freq: dict[str, int] = {}
    for tokens in token_lists:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    for word, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        if cnt >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def make_embedding_layer(matrix: np.ndarray, freeze: bool = True) -> nn.Embedding:
    weight = torch.tensor(matrix, dtype=torch.float32)
    emb = nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=0)
    return emb
