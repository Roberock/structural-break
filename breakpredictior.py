"""
structural_break_pipeline.py

Standalone training pipeline for the CrunchDAO Structural Break competition.

Steps:
1. Load train/test parquet files
2. Train a Sequence VAE to embed variable-length sequences
3. Extract id-level embeddings
4. Train a classifier for structural break detection
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


# ------------------------
# Utils
# ------------------------
def df_to_sequences(X: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None):
    """Convert MultiIndex DataFrame (id, time) into list of arrays [T, D]."""
    if feature_cols is None:
        feature_cols = list(X.columns)
    seqs, ids, lengths = [], [], []
    for gid, g in X.groupby(level="id", sort=True):
        arr = g.droplevel("id").sort_index()[feature_cols].to_numpy(dtype=np.float32)
        seqs.append(arr)
        ids.append(int(gid))
        lengths.append(arr.shape[0])
    return seqs, ids, lengths


def collate_varlen(batch: List[torch.Tensor]):
    """Collate a list of [T_i, D] into padded [B, T_max, D] with mask."""
    lengths = torch.tensor([b.shape[0] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(batch, batch_first=True)  # [B, T_max, D]
    mask = torch.arange(x_pad.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return x_pad, mask, lengths


def sinusoidal_time_emb(T: int, d_model: int, device):
    """Standard transformer-style sinusoidal encoding [T, d_model]."""
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ------------------------
# Variational Sequence Encoder
# ------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.mu = nn.Linear(2 * hidden, latent_dim)
        self.logvar = nn.Linear(2 * hidden, latent_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
        return self.mu(h_last), self.logvar(h_last)


class TimeMLPDecoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int, time_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, T_max, mask):
        B, Z = z.shape
        device = z.device
        time_pe = sinusoidal_time_emb(T_max, self.time_dim, device).unsqueeze(0).expand(B, T_max, -1)
        z_rep = z.unsqueeze(1).expand(B, T_max, -1)
        return self.net(torch.cat([z_rep, time_pe], dim=-1))


class SequenceVAE(nn.Module):
    def __init__(self, in_dim, latent_dim=32, enc_hidden=128, dec_hidden=256, time_dim=32, beta=1.0):
        super().__init__()
        self.encoder = BiLSTMEncoder(in_dim, enc_hidden, latent_dim)
        self.decoder = TimeMLPDecoder(in_dim, latent_dim, time_dim, dec_hidden)
        self.beta = beta
        self.recon_loss = nn.MSELoss(reduction="none")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, mask, lengths):
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)
        T_max = x.size(1)
        x_hat = self.decoder(z, T_max, mask)
        return x_hat, mu, logvar

    def elbo(self, x, mask, lengths):
        x_hat, mu, logvar = self.forward(x, mask, lengths)
        rec = self.recon_loss(x_hat, x).mean(dim=-1)  # [B, T]
        rec = (rec * mask.float()).sum() / mask.sum().clamp_min(1.0)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return rec + self.beta * kl


# ------------------------
# Pipeline class
# ------------------------
class StructuralBreakPipeline:
    def __init__(self, data_dir="data", resources_dir="resources", latent_dim=48):
        self.DATA_DIR = Path(data_dir)
        self.RESOURCES_DIR = Path(resources_dir)
        self.RESOURCES_DIR.mkdir(exist_ok=True)
        self.latent_dim = latent_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vae = None
        self.classifier = None

    def load_data(self):
        X_train = pd.read_parquet(self.DATA_DIR / "X_train.parquet")
        y_train = pd.read_parquet(self.DATA_DIR / "y_train.parquet").squeeze()
        X_test = pd.read_parquet(self.DATA_DIR / "X_test.reduced.parquet")
        y_test = pd.read_parquet(self.DATA_DIR / "y_test.reduced.parquet").squeeze()
        return X_train, y_train, X_test, y_test

    def train_vae(self, X, feature_cols=None, epochs=5, batch_size=64, lr=1e-3):
        seqs, ids, lengths = df_to_sequences(X, feature_cols)
        in_dim = seqs[0].shape[1]

        class SeqDataset(Dataset):
            def __init__(self, seqs): self.data = [torch.tensor(s) for s in seqs]
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return self.data[i]

        dl = DataLoader(SeqDataset(seqs), batch_size=batch_size, shuffle=True, collate_fn=collate_varlen)

        self.vae = SequenceVAE(in_dim, latent_dim=self.latent_dim).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            print(f'epoch - {ep}')
            self.vae.train()
            losses = []
            for x_pad, mask, lengths in dl:
                x_pad, mask, lengths = x_pad.to(self.device), mask.to(self.device), lengths.to(self.device)
                loss = self.vae.elbo(x_pad, mask, lengths)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print(f"[VAE epoch {ep}] loss={np.mean(losses):.4f}")

    @torch.no_grad()
    def embed(self, X, feature_cols=None, batch_size=128):
        seqs, ids, _ = df_to_sequences(X, feature_cols)
        class SeqDataset(Dataset):
            def __init__(self, seqs): self.data = [torch.tensor(s) for s in seqs]
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return self.data[i]
        dl = DataLoader(SeqDataset(seqs), batch_size=batch_size, shuffle=False, collate_fn=collate_varlen)

        self.vae.eval().to(self.device)
        mu_list = []
        for x_pad, mask, lengths in dl:
            x_pad, mask, lengths = x_pad.to(self.device), mask.to(self.device), lengths.to(self.device)
            mu, logvar = self.vae.encoder(x_pad, lengths)
            mu_list.append(mu.cpu())
        Z = torch.cat(mu_list).numpy()
        return pd.DataFrame(Z, index=pd.Index(ids, name="id"), columns=[f"z_{i}" for i in range(Z.shape[1])])

    def train_classifier(self, X_emb, y):
        clf = LogisticRegression(max_iter=10000, class_weight="balanced")
        clf.fit(X_emb.loc[y.index], y.values)
        self.classifier = clf
        scores = clf.predict_proba(X_emb)[:, 1]
        print("Train ROC-AUC:", roc_auc_score(y.loc[X_emb.index], scores))

    def save(self):
        joblib.dump(self.classifier, self.RESOURCES_DIR / "classifier.joblib")
        torch.save(self.vae.state_dict(), self.RESOURCES_DIR / "vae.pt")
        print("[Pipeline] Saved classifier + VAE state.")

    def run(self, epochs=5):
        X_train, y_train, X_test, y_test = self.load_data()
        self.train_vae(X_train, epochs=epochs)
        X_emb = self.embed(X_train)
        self.train_classifier(X_emb, y_train)
        self.save()
        # Prepare test embeddings for submission
        X_test_emb = self.embed(X_test)
        X_test_emb.to_csv(self.RESOURCES_DIR / "X_test_embeddings.csv")
        print("[Pipeline] Test embeddings saved.")


if __name__ == "__main__":
    pipe = StructuralBreakPipeline()
    pipe.run(epochs=5)