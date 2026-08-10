"""Optional deep-learning module (PyTorch).

Implements a small character-level sequence classifier with configurable
architecture: LSTM, BiLSTM or GRU followed by a dense head. It is fully
optional: everything is guarded so importing this module works even when
PyTorch is not installed (``available()`` returns ``False`` then).

This is a deliberately lightweight reference implementation - the classical
ML pipeline in :mod:`src.train` is the recommended path because it is faster
to train and reaches >99% accuracy on this task.
"""

from __future__ import annotations

from src.utils import get_logger

logger = get_logger("deep_learning")


def available() -> bool:
    """Return ``True`` if PyTorch is installed."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


if available():
    import numpy as np
    import torch
    import torch.nn as nn

    class CharLevelSequenceClassifier(nn.Module):
        """Character-level text classifier with LSTM / BiLSTM / GRU backbone.

        Args:
            vocab_size: Number of characters in the vocabulary.
            hidden_size: Hidden dimension of the recurrent layer.
            n_classes: Number of output languages.
            arch: One of ``"lstm"``, ``"bilstm"``, ``"gru"``.
            num_layers: Stacked recurrent layers.
            dropout: Dropout probability between layers / in the head.
            embedding_dim: Character embedding dimension.
        """

        def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            n_classes: int,
            arch: str = "bilstm",
            num_layers: int = 2,
            dropout: float = 0.3,
            embedding_dim: int = 64,
        ) -> None:
            super().__init__()
            if arch not in {"lstm", "bilstm", "gru"}:
                raise ValueError(f"Unsupported arch {arch!r}")

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

            rnn_cls = nn.LSTM if arch in {"lstm", "bilstm"} else nn.GRU
            bidirectional = arch == "bilstm"
            self.rnn = rnn_cls(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            direction = 2 if bidirectional else 1
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * direction, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, n_classes),
            )

        def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                input_ids: ``(batch, seq_len)`` LongTensor of char ids.
                lengths: ``(batch,)`` sequence lengths (pre-padding).

            Returns:
                Logits of shape ``(batch, n_classes)``.
            """
            x = self.embedding(input_ids)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

            # Mean-pool the padded output (mask pads).
            mask = input_ids.ne(0).unsqueeze(-1).float()
            summed = (out * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            pooled = summed / count
            return self.head(pooled)

    def train_sequence_classifier(
        X_texts: list[str],
        y: np.ndarray,
        n_classes: int,
        arch: str = "bilstm",
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        hidden_size: int = 128,
        max_len: int = 128,
        device: str = "cpu",
    ) -> "CharLevelSequenceClassifier":
        """Train a character-level sequence classifier.

        Args:
            X_texts: Raw (uncleaned) texts.
            y: Integer labels.
            n_classes: Number of classes.
            arch: Recurrent architecture.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            hidden_size: Recurrent hidden size.
            max_len: Maximum character length (padding/truncation).
            device: ``"cpu"`` or ``"cuda"``.

        Returns:
            The trained model (on CPU).
        """
        import torch

        from src.preprocessing import clean_text

        texts = [clean_text(t)[:max_len] or " " for t in X_texts]
        vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        for text in texts:
            for ch in text:
                vocab.setdefault(ch, len(vocab))

        def encode(text: str) -> tuple[list[int], int]:
            ids = [vocab.get(ch, 1) for ch in text]
            return ids, len(ids)

        enc = [encode(t) for t in texts]
        X_ids = np.zeros((len(enc), max_len), dtype=np.int64)
        X_len = np.zeros(len(enc), dtype=np.int64)
        for i, (ids, ln) in enumerate(enc):
            X_ids[i, :ln] = ids[:max_len]
            X_len[i] = ln

        model = CharLevelSequenceClassifier(
            vocab_size=len(vocab),
            hidden_size=hidden_size,
            n_classes=n_classes,
            arch=arch,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_ids), torch.tensor(X_len), torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(1, epochs + 1):
            total_loss, n_batches = 0.0, 0
            for ids, ln, labels in loader:
                ids, ln, labels = ids.to(device), ln.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(ids, ln)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            logger.info("Epoch %2d/%d loss=%.4f", epoch, epochs, total_loss / max(n_batches, 1))

        return model.cpu() if device != "cpu" else model


if __name__ == "__main__":
    print(f"PyTorch available: {available()}")
    if not available():
        print("Install PyTorch to use the optional deep-learning module.")
