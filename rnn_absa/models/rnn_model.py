"""
Unified RNN model supporting: RNN, LSTM, GRU (unidirectional & bidirectional).
Architecture:
  Embedding → RNN/LSTM/GRU (n layers, optional bidirectional) → last hidden → Dropout → Linear
"""

import torch
import torch.nn as nn


ARCH_MAP = {
    'rnn':    (nn.RNN,  False),
    'lstm':   (nn.LSTM, False),
    'gru':    (nn.GRU,  False),
    'birnn':  (nn.RNN,  True),
    'bilstm': (nn.LSTM, True),
    'bigru':  (nn.GRU,  True),
}


class RNNClassifier(nn.Module):
    def __init__(
        self,
        arch: str,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        embedding_weight: torch.Tensor = None,
        freeze_embedding: bool = True,
        pad_idx: int = 0,
    ):
        super().__init__()
        assert arch in ARCH_MAP, f"Unknown arch: {arch}. Choose from {list(ARCH_MAP)}"
        rnn_cls, bidirectional = ARCH_MAP[arch]
        self.arch          = arch
        self.bidirectional = bidirectional
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.is_lstm       = (rnn_cls == nn.LSTM)

        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weight, freeze=freeze_embedding, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        directions   = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) — token indices
        emb = self.embedding(x)              # (batch, seq, embed_dim)

        if self.is_lstm:
            out, (hn, _) = self.rnn(emb)
        else:
            out, hn = self.rnn(emb)
        # hn: (num_layers * directions, batch, hidden_size)

        if self.bidirectional:
            # concat last layer forward + backward hidden
            fwd = hn[-2]   # (batch, hidden_size)
            bwd = hn[-1]   # (batch, hidden_size)
            h   = torch.cat([fwd, bwd], dim=-1)
        else:
            h = hn[-1]     # (batch, hidden_size)

        h   = self.dropout(h)
        out = self.fc(h)   # (batch, num_classes)
        return out
