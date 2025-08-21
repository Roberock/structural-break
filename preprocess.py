from pathlib import Path
import pandas as pd
from typing import List, Sequence, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# train_vae_features.py
from typing import Optional, Sequence, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("data/")
RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)


def load_data(data_path = None):
    if data_path is None:
        DATA_DIR = Path("data")
    else:
        DATA_DIR = Path(data_path)
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.reduced.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(DATA_DIR / "y_test.reduced.parquet").squeeze()
    return X_train, X_test, y_train, y_test

def data_exploration():
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")

    # Count sequences
    n_ids = X_train.index.get_level_values("id").nunique()
    print(f"Number of sequences: {n_ids}")

    # Distribution of sequence lengths
    counts = X_train.groupby(level="id").size()
    print("Sequence length distribution:")
    print(counts.describe())

    # Target balance
    y_train = y_train.squeeze()
    print("Target distribution:")
    print(y_train.value_counts())

    return X_train, y_train


def X_to_wide(X: pd.DataFrame, value_col="value", period_col="period") -> pd.DataFrame:
    """
    Transform long-format X (indexed by id,time) into wide-format with
    v_t{t} and p_t{t} columns.

    Parameters
    ----------
    X : DataFrame with MultiIndex (id,time) and columns [value, period]
    value_col : str, default 'value'
    period_col: str, default 'period'

    Returns
    -------
    DataFrame indexed by id, wide features with NaNs for missing times.
    """
    # make sure we have id,time as index
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("X must have MultiIndex (id,time)")

    # reset so we can pivot
    df = X.reset_index()

    # pivot for values
    v_wide = df.pivot(index="id", columns="time", values=value_col)
    v_wide = v_wide.add_prefix("v_t")

    # pivot for periods
    p_wide = df.pivot(index="id", columns="time", values=period_col)
    p_wide = p_wide.add_prefix("p_t")

    # combine
    Z = pd.concat([v_wide, p_wide], axis=1).sort_index(axis=1)
    Z.index.name = "id"

    return Z


def build_features(X):
    features = X.groupby("id").agg({
        "value": ["mean", "std", "min", "max"],
        "period": ["nunique"]
    })
    features.columns = ["_".join(col) for col in features.columns]
    return features


def df_to_sequences(
    X: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Convert a MultiIndex (id, time) DataFrame into a list of variable-length arrays.
    Returns:
        seqs: list of arrays [T_i, D]
        ids:  list of ids aligned with seqs
        lengths: list of lengths T_i
    """
    if feature_cols is None:
        feature_cols = list(X.columns)
    # Keep only requested features (this makes it 'any-dim X')
    Xf = X[feature_cols].copy()

    seqs, ids, lengths = [], [], []
    for gid, g in Xf.groupby(level='id', sort=True):
        arr = g.droplevel('id').sort_index().to_numpy(dtype=np.float32)  # [T, D]
        seqs.append(arr)
        ids.append(int(gid))
        lengths.append(arr.shape[0])
    return seqs, ids, lengths



def sinusoidal_time_emb(T: int, d_model: int, device):
    """[T, d_model] standard transformer-style positional encodings."""
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def collate_varlen(
    batch: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of [T_i, D] into:
        x_pad [B, T_max, D], mask [B, T_max] (True=valid), lengths [B]
    """
    lengths = torch.tensor([b.shape[0] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(batch, batch_first=True)  # [B, T_max, D]
    # mask True on valid positions
    mask = torch.arange(x_pad.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return x_pad, mask, lengths

class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.mu = nn.Linear(2*hidden, latent_dim)
        self.logvar = nn.Linear(2*hidden, latent_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, T, D]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h_n, c_n) = self.rnn(packed)
        # concat last layer forward/backward hidden: [2, B, H] -> [B, 2H]
        h_last = torch.concat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last)
        return mu, logvar

class TimeMLPDecoder(nn.Module):
    """
    Decoder that reconstructs x_t from z and a sinusoidal time embedding.
    """
    def __init__(self, out_dim: int, latent_dim: int, time_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z: torch.Tensor, T_max: int, mask: torch.Tensor):
        """
        z: [B, Z]
        returns x_hat: [B, T_max, D]
        """
        B = z.size(0)
        device = z.device
        time_pe = sinusoidal_time_emb(T_max, self.time_dim, device=device)  # [T_max, time_dim]
        time_pe = time_pe.unsqueeze(0).expand(B, T_max, self.time_dim)      # [B, T_max, time_dim]
        z_rep = z.unsqueeze(1).expand(B, T_max, z.size(-1))                 # [B, T_max, Z]
        inp = torch.cat([z_rep, time_pe], dim=-1)                           # [B, T_max, Z+time_dim]
        return self.net(inp)                                                # [B, T_max, D]

class SequenceVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 32,
        enc_hidden: int = 128,
        enc_layers: int = 1,
        dec_hidden: int = 256,
        time_dim: int = 32,
        beta: float = 1.0,
    ):
        super().__init__()
        self.encoder = BiLSTMEncoder(in_dim, enc_hidden, latent_dim, num_layers=enc_layers)
        self.decoder = TimeMLPDecoder(out_dim=in_dim, latent_dim=latent_dim, time_dim=time_dim, hidden=dec_hidden)
        self.beta = beta
        self.recon_loss = nn.MSELoss(reduction='none')

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, mask, lengths):
        # encode
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        # decode to full T_max, then mask loss
        T_max = x.size(1)
        x_hat = self.decoder(z, T_max, mask)
        return x_hat, mu, logvar

    def elbo(self, x, mask, lengths):
        x_hat, mu, logvar = self.forward(x, mask, lengths)
        # recon loss over valid positions
        rec = self.recon_loss(x_hat, x).mean(dim=-1)        # [B, T]
        rec = (rec * mask.float()).sum() / mask.sum().clamp_min(1.0)
        # KL(q||p) with p=N(0,I)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        loss = rec + self.beta * kl
        return loss, {'rec': rec.item(), 'kl': kl.item()}

    @torch.no_grad()
    def encode(self, x, mask, lengths):
        mu, logvar = self.encoder(x, lengths)
        return mu  # use μ as deterministic embedding






"""class SeqDataset(Dataset):
    def __init__(self, seqs: Sequence[np.ndarray]):
        self.data = [torch.from_numpy(s) for s in seqs]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def train_vae_on_X(
    X_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    latent_dim: int = 48,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SequenceVAE:
    seqs, ids, lengths = df_to_sequences(X_df, feature_cols=feature_cols)
    in_dim = seqs[0].shape[1]
    ds = SeqDataset(seqs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_varlen, drop_last=False)

    model = SequenceVAE(in_dim=in_dim, latent_dim=latent_dim, beta=beta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs+1):
        tot = 0.0
        for batch in dl:
            x_pad, mask, lengths = batch
            x_pad = x_pad.to(device)            # [B, T, D]
            mask = mask.to(device)              # [B, T]
            lengths = lengths.to(device)
            loss, logs = model.elbo(x_pad, mask, lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * x_pad.size(0)
        print(f"[ep {ep:03d}] loss={tot/len(ds):.4f} rec={logs['rec']:.4f} kl={logs['kl']:.4f}")
    return model

@torch.no_grad()
def vae_embeddings(
    model: SequenceVAE,
    X_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    seqs, ids, lengths = df_to_sequences(X_df, feature_cols=feature_cols)
    ds = SeqDataset(seqs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_varlen, drop_last=False)
    model.eval().to(device)

    mu_list = []
    ptr = 0
    for x_pad, mask, lens in dl:
        x_pad = x_pad.to(device); mask = mask.to(device); lens = lens.to(device)
        mu = model.encode(x_pad, mask, lens)     # [B, Z]
        mu_list.append(mu.cpu())
        ptr += x_pad.size(0)

    Z = torch.cat(mu_list, dim=0).numpy()
    emb = pd.DataFrame(Z, index=pd.Index(ids, name="id"))
    emb.columns = [f"z_{i}" for i in range(emb.shape[1])]
    return emb
"""