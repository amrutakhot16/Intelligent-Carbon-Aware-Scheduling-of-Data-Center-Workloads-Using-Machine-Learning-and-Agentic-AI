"""
Step 10c: Model 5 — GRU (gated recurrent unit network).

Same input, output and training recipe as the LSTM, so the two are a fair
comparison. A GRU has two gates instead of the LSTM's three and no separate cell state: fewer parameters,
usually faster to train, and often just as accurate on short sequences.
"""
import torch
import torch.nn as nn

from settings import DL
from models.dl_common import SeqData, fit


class GRUForecaster(nn.Module):
    """
    GRU over the last seq_len hours of features. Each region gets a learned embedding
    vector, which is added to the input at every time step and again before the output
    layer, so the model can behave differently for each grid.
    """
    def __init__(self, n_features: int, n_regions: int, cfg: dict = DL):
        """
        :param n_features: number of input features per hour
        :param n_regions: number of regions (size of the embedding table)
        :param cfg: hyper-parameters from config.json (hidden, region_emb, num_layers, dropout)
        """
        super().__init__()
        h, e, layers = cfg["hidden"], cfg["region_emb"], cfg["num_layers"]
        self.emb = nn.Embedding(n_regions, e)
        self.rnn = nn.GRU(n_features + e, h, num_layers=layers, batch_first=True,
                          dropout=cfg["dropout"] if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(h + e, h), nn.ReLU(), nn.Dropout(cfg["dropout"]), nn.Linear(h, 1))

    def forward(self, x: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
        """
        :param x: input windows, shape (batch, seq_len, n_features)
        :param reg: region ids, shape (batch,)
        :return: predicted scaled change from 24 hours earlier, shape (batch,)
        """
        e = self.emb(reg)                                                  # (B, E)
        seq = torch.cat([x, e.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        out, _ = self.rnn(seq)                                             # (B, L, H)
        return self.head(torch.cat([out[:, -1], e], dim=-1)).squeeze(-1)


def train_gru(seq: SeqData, fast_mode: bool = True):
    """
    Build and train the GRU with the shared training loop in dl_common.fit().

    :param seq: sequence data from dl_common.build_sequence_data()
    :param fast_mode: fewer epochs and a random subset of rows per epoch
    :return: (network, train predictions, test predictions, info)
    """
    return fit(GRUForecaster(seq.n_features, seq.n_regions), seq, "GRU", fast_mode)
