"""
Step 10d: Model 6 — compact Temporal Fusion Transformer (TFT-style).

A from-scratch PyTorch version of the main TFT building blocks (Lim et al., 2021), sized for this project:
* static input        : the region (embedding) -> context vectors that steer the rest of the network
* variable selection  : a learned, per-hour weight for each of the numeric features (we can read these weights back)
* LSTM encoder        : local, sequential processing of the window
* gated residual nets : gates that let the network skip layers it does not need
* self-attention      : the last hour attends over the whole window (which past hours matter?)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from settings import DL, FIG_DIR, NUM_FEATURES
from models.dl_common import SeqData, fit


class GRN(nn.Module):
    """
    Gated residual network: Linear -> ELU -> Linear -> GLU gate, added to a skip connection
    and layer-normalized. An optional context vector (here the region) can be added after
    the first layer.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1, d_ctx: int = 0):
        """
        :param d_in: input size
        :param d_hidden: hidden layer size
        :param d_out: output size
        :param dropout: dropout rate before the gate
        :param d_ctx: size of the optional context vector (0 = no context)
        """
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, 2 * d_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: input, shape (batch, d_in) or (batch, hours, d_in)
        :param ctx: optional context vector (region), shape (batch, d_ctx)
        :return: output, same leading shape as x with last size d_out
        """
        h = self.fc1(x)
        if self.ctx is not None and ctx is not None:
            h = h + self.ctx(ctx.unsqueeze(1) if ctx.dim() == 2 and x.dim() == 3 else ctx)
        h = self.fc2(F.elu(h))
        h = F.glu(self.gate(self.drop(h)), dim=-1)
        return self.norm(self.skip(x) + h)


class GateAddNorm(nn.Module):
    """
    Gate the output of a layer, add it to a skip connection and normalize.
    """
    def __init__(self, d: int, dropout: float = 0.1):
        """
        :param d: size of the input and output
        :param dropout: dropout rate before the gate
        """
        super().__init__()
        self.drop, self.gate, self.norm = nn.Dropout(dropout), nn.Linear(d, 2 * d), nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        :param x: output of the previous layer
        :param skip: input to add back (same shape as x)
        :return: normalized sum of the gated x and skip
        """
        return self.norm(skip + F.glu(self.gate(self.drop(x)), dim=-1))


class VariableSelection(nn.Module):
    """
    Turn each numeric feature into its own vector, then combine them with softmax weights
    learned per sample and per hour. The weights show how much the model uses each feature.
    """
    def __init__(self, n_vars: int, d: int, dropout: float, d_ctx: int):
        """
        :param n_vars: number of numeric features
        :param d: vector size per feature
        :param dropout: dropout rate inside the weight network
        :param d_ctx: size of the context vector (region)
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_vars, d))
        self.select = GRN(n_vars * d, d, n_vars, dropout, d_ctx)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor):
        """
        :param x: input windows, shape (batch, hours, n_vars)
        :param ctx: region context vector, shape (batch, d_ctx)
        :return: (combined vector per hour (batch, hours, d), feature weights (batch, hours, n_vars))
        """
        emb = x.unsqueeze(-1) * self.w + self.b                          # (B, L, V, D)
        weights = torch.softmax(self.select(emb.flatten(-2), ctx), dim=-1)   # (B, L, V)
        return (emb * weights.unsqueeze(-1)).sum(-2), weights            # (B, L, D), (B, L, V)


class TFTForecaster(nn.Module):
    """
    The full model: region context -> variable selection -> LSTM -> enrichment with the
    region context -> attention from the last hour over the window -> output layer.
    """
    def __init__(self, n_features: int, n_regions: int, cfg: dict = DL):
        """
        :param n_features: number of input features per hour
        :param n_regions: number of regions (size of the embedding table)
        :param cfg: hyper-parameters from config.json (tft_d_model, tft_heads, dropout)
        """
        super().__init__()
        d, heads, p = cfg["tft_d_model"], cfg["tft_heads"], cfg["dropout"]
        self.static_emb = nn.Embedding(n_regions, d)
        self.ctx_select, self.ctx_h, self.ctx_c, self.ctx_enrich = (GRN(d, d, d, p) for _ in range(4))
        self.vsn = VariableSelection(n_features, d, p, d_ctx=d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1 = GateAddNorm(d, p)
        self.enrich = GRN(d, d, d, p, d_ctx=d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.gate2 = GateAddNorm(d, p)
        self.ff = GRN(d, d, d, p)
        self.gate3 = GateAddNorm(d, p)
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, reg: torch.Tensor, return_weights: bool = False):
        """
        :param x: input windows, shape (batch, seq_len, n_features)
        :param reg: region ids, shape (batch,)
        :param return_weights: also return the variable selection weights
        :return: predicted scaled change from 24 hours earlier, shape (batch,),
                 plus the weights (batch, seq_len, n_features) if return_weights is True
        """
        s = self.static_emb(reg)                                          # (B, D) static context from the region
        z, w = self.vsn(x, self.ctx_select(s))                            # which features matter, hour by hour
        h0, c0 = self.ctx_h(s).unsqueeze(0).contiguous(), self.ctx_c(s).unsqueeze(0).contiguous()
        enc, _ = self.lstm(z, (h0, c0))                                   # sequential processing
        h = self.enrich(self.gate1(enc, z), self.ctx_enrich(s))           # (B, L, D)
        q = h[:, -1:]                                                     # the target hour asks the whole window
        a, _ = self.attn(q, h, h, need_weights=False)
        h = self.gate3(self.ff(self.gate2(a, q)), q)                      # (B, 1, D)
        y = self.out(h.squeeze(1)).squeeze(-1)
        return (y, w) if return_weights else y


def train_tft(seq: SeqData, fast_mode: bool = True):
    """
    Build and train the TFT with the shared training loop in dl_common.fit().

    :param seq: sequence data from dl_common.build_sequence_data()
    :param fast_mode: fewer epochs and a random subset of rows per epoch
    :return: (network, train predictions, test predictions, info)
    """
    return fit(TFTForecaster(seq.n_features, seq.n_regions), seq, "TFT", fast_mode)


@torch.no_grad()
def variable_importance(model: TFTForecaster, seq: SeqData, n_rows: int = 20000) -> pd.Series:
    """
    Average variable selection weight of each numeric feature, over a random sample of
    test rows and all hours in their windows.

    :param model: trained TFT
    :param seq: sequence data
    :param n_rows: number of test rows to sample
    :return: average weight per feature, sorted from smallest to largest
    """
    model.eval()
    rows = np.where(~seq.is_val & ~seq.is_fit)[0]                         # test rows only
    rows = np.random.default_rng(0).choice(rows, min(n_rows, len(rows)), replace=False)
    w = [model(*seq.batch(rows[i:i + 2048])[:2], return_weights=True)[1].mean(1).cpu().numpy()
         for i in range(0, len(rows), 2048)]
    return pd.Series(np.concatenate(w).mean(0), index=NUM_FEATURES).sort_values()


def plot_variable_importance(model: TFTForecaster, seq: SeqData, top_n: int = 15, fig_dir=FIG_DIR) -> pd.Series:
    """
    Bar chart of the features with the highest average selection weight.

    :param model: trained TFT
    :param seq: sequence data
    :param top_n: number of features to show
    :param fig_dir: folder to save the figure in
    :return: average weight per feature (all features, sorted ascending)
    """
    imp = variable_importance(model, seq)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.iloc[-top_n:].plot.barh(ax=ax, color="#3a9d5d")
    ax.set(title=f"TFT — top {top_n} features by selection weight (2024 test rows)", xlabel="average weight")
    plt.tight_layout(); plt.savefig(fig_dir / "tft_variable_importance.png", dpi=120); plt.close()
    return imp
