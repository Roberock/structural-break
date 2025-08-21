import numpy as np
import pandas as pd


def _safe_series_stats(s: pd.Series, prefix: str) -> dict:
    """Basic stats with NaN safety."""
    d = {}
    if len(s) == 0:
        keys = ["mean","std","min","max","median","q10","q90","skew","kurt"]
        return {f"{prefix}_{k}": 0.0 for k in keys}
    d[f"{prefix}_mean"]   = float(s.mean())
    d[f"{prefix}_std"]    = float(s.std(ddof=1)) if s.size > 1 else 0.0
    d[f"{prefix}_min"]    = float(s.min())
    d[f"{prefix}_max"]    = float(s.max())
    d[f"{prefix}_median"] = float(s.median())
    q = s.quantile([0.10, 0.90])
    d[f"{prefix}_q10"]    = float(q.loc[0.10])
    d[f"{prefix}_q90"]    = float(q.loc[0.90])
    # pandas has skew/kurt; guard very short series
    d[f"{prefix}_skew"]   = float(s.skew())  if s.size > 2 else 0.0
    d[f"{prefix}_kurt"]   = float(s.kurt())  if s.size > 3 else 0.0
    return d

def _ewm_tail_feats(s: pd.Series, alphas=(0.1, 0.01, 0.001), base_name="value"):
    """Exponentially-weighted moving stats (last value), fixed-size regardless of T."""
    out = {}
    for a in alphas:
        m = s.ewm(alpha=a, adjust=False).mean()
        v = s.ewm(alpha=a, adjust=False).var(bias=False)
        out[f"ewm{a:g}_mean_last"] = float(m.iloc[-1])
        out[f"ewm{a:g}_std_last"]  = float(np.sqrt(v.iloc[-1])) if not np.isnan(v.iloc[-1]) else 0.0
    return out

def _ensure_dataframe_index(df_like: dict, index: pd.Index) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(df_like, orient="index")
    df.index.name = "id"
    df = df.reindex(index)  # align order to ids encountered
    return df

# 1) Fourier features
def f_x2z_fourier(X: pd.DataFrame, Nf: int = 10, value_col: str = "value") -> pd.DataFrame:
    """
    Per-id power in the first Nf rFFT bins (excluding DC), normalized by total power.
    Returns a DataFrame indexed by id with columns fpow_1..fpow_Nf.
    """
    feats = {}
    for gid, g in X.groupby(level="id", sort=True):
        v = g[value_col].to_numpy(dtype=np.float64)
        if v.size < 2:
            # degenerate case -> zeros
            feats[int(gid)] = {f"fpow_{k}": 0.0 for k in range(1, Nf+1)}
            continue
        v = v - v.mean()
        spec = np.fft.rfft(v)                     # [T//2+1]
        power = np.abs(spec)**2
        # exclude DC (bin 0), take next Nf bins
        band = power[1:1+Nf]
        # pad if too short
        if band.size < Nf:
            band = np.pad(band, (0, Nf - band.size), constant_values=0.0)
        total = power.sum()
        if total <= 0:
            norm = np.zeros(Nf, dtype=np.float64)
        else:
            norm = band / total
        feats[int(gid)] = {f"fpow_{k}": float(norm[k-1]) for k in range(1, Nf+1)}
    return pd.DataFrame.from_dict(feats, orient="index").rename_axis("id")

# 2) Moment / segment features
def f_x2z_moments(X: pd.DataFrame, value_col: str = "value", period_col: str = "period") -> pd.DataFrame:
    """
    Fixed-size, per-id features capturing whole/pre/post segment stats, deltas,
    counts, and EWM tails.
    """
    rows = {}
    for gid, g in X.groupby(level="id", sort=True):
        s = g[value_col].astype(float)
        p = g[period_col].astype(int) if period_col in g.columns else pd.Series(np.zeros_like(s), index=s.index)

        whole = _safe_series_stats(s, "whole")
        pre   = _safe_series_stats(s[p == 0], "pre")
        post  = _safe_series_stats(s[p == 1], "post")

        # deltas post - pre for a few key stats (robust shift features)
        delta = {
            "delta_mean":  post["post_mean"] - pre["pre_mean"],
            "delta_std":   post["post_std"]  - pre["pre_std"],
            "delta_median":post["post_median"] - pre["pre_median"],
            "delta_q10":   post["post_q10"]  - pre["pre_q10"],
            "delta_q90":   post["post_q90"]  - pre["pre_q90"],
            "delta_skew":  post["post_skew"] - pre["pre_skew"],
            "delta_kurt":  post["post_kurt"] - pre["pre_kurt"],
        }

        # lengths
        n_pre  = int((p == 0).sum())
        n_post = int((p == 1).sum())
        counts = {
            "n_pre": n_pre,
            "n_post": n_post,
            "n_total": int(len(s)),
            "post_pre_ratio": float(n_post / n_pre) if n_pre > 0 else 0.0
        }

        # EWM tails (length-invariant)
        ewm_feats = _ewm_tail_feats(s, alphas=(0.1, 0.01, 0.001))

        row = {}
        row.update(whole); row.update(pre); row.update(post)
        row.update(delta); row.update(counts); row.update(ewm_feats)
        rows[int(gid)] = row

    df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("id")
    # Safety: fill any NaNs (can happen for tiny segments)
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

# 3) Expert combo
def f_x2z_expert(X: pd.DataFrame, Nf: int = 10, value_col: str = "value", period_col: str = "period") -> pd.DataFrame:
    """
    Concatenate moments/segment features with Fourier band powers.
    """
    A = f_x2z_moments(X, value_col=value_col, period_col=period_col)
    B = f_x2z_fourier(X, Nf=Nf, value_col=value_col)
    Z = A.join(B, how="left").fillna(0.0)
    return Z

# 4) VAE embeddings (optional)
def f_x2z_vae(
    X: pd.DataFrame,
    vae_encode_fn=None,
    feature_cols: list | None = None
) -> pd.DataFrame:
    """
    Variational encoder interface.
    - If you already trained a VAE, pass a callable `vae_encode_fn(X_df) -> DataFrame`
      that returns id-indexed embeddings (e.g., the μ vectors).
    - Otherwise, fall back to expert features.
    """
    if callable(vae_encode_fn):
        Z = vae_encode_fn(X if feature_cols is None else X[feature_cols])
        # Ensure index is 'id' and type is DataFrame
        if not isinstance(Z, pd.DataFrame):
            raise ValueError("vae_encode_fn must return a pandas.DataFrame")
        if Z.index.name != "id":
            Z.index.name = "id"
        return Z
    # Fallback
    return f_x2z_expert(X)




import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import joblib

RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)

# -------------------------------
# FAST FEATURE EXTRACTORS
# -------------------------------
def _agg_basic_stats(df, value_col):
    agg = {
        value_col: [
            "mean", "std", "min", "max", "median",
            lambda s: s.quantile(0.10),
            lambda s: s.quantile(0.90),
            "skew", "kurt"
        ]
    }
    g = df.groupby(level="id")[value_col].agg(agg[value_col])
    g.columns = [
        "mean","std","min","max","median","q10","q90","skew","kurt"
    ]
    return g.add_prefix("whole_")

def _agg_segment_stats(df, value_col, period_col, seg):
    """seg=0 or 1 (pre/post)."""
    sub = df[df[period_col] == seg]
    if sub.empty:
        # no rows for this segment
        return pd.DataFrame(index=df.index.get_level_values("id").unique()).assign(
            **{f"{'pre' if seg==0 else 'post'}_{n}": 0.0 for n in
               ["mean","std","min","max","median","q10","q90","skew",""]}
        )
    agg = sub.groupby(level="id")[value_col].agg([
        "mean","std","min","max","median",
        lambda s: s.quantile(0.10),
        lambda s: s.quantile(0.90),
        "skew","kurt"
    ])
    agg.columns = ["mean","std","min","max","median","q10","q90","skew","kurt"]
    return agg.add_prefix("pre_" if seg == 0 else "post_")

def _length_feats(df, period_col):
    g = df.groupby(level="id")[period_col]
    n_total = g.size().rename("n_total")
    # counts by value
    counts = g.value_counts().unstack(fill_value=0)
    counts = counts.rename(columns={0: "n_pre", 1: "n_post"})
    if "n_pre" not in counts: counts["n_pre"] = 0
    if "n_post" not in counts: counts["n_post"] = 0
    out = pd.concat([n_total, counts[["n_pre","n_post"]]], axis=1).fillna(0)
    out["post_pre_ratio"] = np.where(out["n_pre"] > 0, out["n_post"] / out["n_pre"], 0.0)
    return out

def _ewm_tail_last(df, value_col, alphas=(0.1, 0.01, 0.001)):
    """Exponentially-weighted stats per id (last value only). Vectorised via groupby.transform + last()."""
    out = pd.DataFrame(index=df.index.get_level_values("id").unique())
    # we need the last index per id to pick tail values
    last_idx = df.groupby(level="id").tail(1).index
    for a in alphas:
        mean_series = df.groupby(level="id")[value_col].transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
        var_series  = df.groupby(level="id")[value_col].transform(lambda s: s.ewm(alpha=a, adjust=False).var(bias=False))
        mean_last = mean_series.loc[last_idx]
        std_last  = var_series.loc[last_idx].clip(lower=0).pow(0.5)
        mean_last.index = mean_last.index.droplevel("time")
        std_last.index  = std_last.index.droplevel("time")
        out[f"ewm{a:g}_mean_last"] = mean_last.astype(float)
        out[f"ewm{a:g}_std_last"]  = std_last.astype(float).fillna(0.0)
    return out

def _fft_band_powers_fast(df, value_col="value", Nf=12):
    """
    Fast-ish rFFT per id by operating on contiguous blocks.
    Assumes df is sorted by (id, time). Uses NumPy to minimise pandas overhead.
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    ids = df.index.get_level_values("id").to_numpy()
    vals = df[value_col].to_numpy(dtype=np.float64)

    # boundaries where id changes
    boundaries = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [len(ids)]])
    out_rows = {}
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        gid = int(ids[b0])
        v = vals[b0:b1]
        if v.size < 2:
            out_rows[gid] = {f"fpow_{k}": 0.0 for k in range(1, Nf+1)}
            continue
        v = v - v.mean()
        power = np.abs(np.fft.rfft(v))**2
        band = power[1:1+Nf]
        if band.size < Nf:
            band = np.pad(band, (0, Nf - band.size))
        total = power.sum()
        norm = (band / total) if total > 0 else np.zeros_like(band)
        out_rows[gid] = {f"fpow_{k}": float(norm[k-1]) for k in range(1, Nf+1)}
    return pd.DataFrame.from_dict(out_rows, orient="index").rename_axis("id").fillna(0.0)


def f_x2z_expert_fast(X: pd.DataFrame, Nf: int = 12,
                      value_col: str = "value", period_col: str = "period") -> pd.DataFrame:
    """Vectorised expert features (fast)."""
    # Ensure sort for FFT block splits
    if not X.index.is_monotonic_increasing:
        X = X.sort_index()

    # Whole / pre / post stats
    whole = _agg_basic_stats(X, value_col=value_col)
    pre   = _agg_segment_stats(X, value_col=value_col, period_col=period_col, seg=0)
    post  = _agg_segment_stats(X, value_col=value_col, period_col=period_col, seg=1)

    # Align frames and fill gaps
    feats = whole.join(pre, how="outer").join(post, how="outer").fillna(0.0)

    # Deltas (post - pre)
    for k in ["mean","std","median","q10","q90","skew"]:
        feats[f"delta_{k}"] = feats.get(f"post_{k}", 0.0) - feats.get(f"pre_{k}", 0.0)

    # Length features + EWM tails
    lenf = _length_feats(X, period_col=period_col)
    ewmf = _ewm_tail_last(X, value_col=value_col, alphas=(0.1, 0.01, 0.001))

    feats = feats.join(lenf, how="left").join(ewmf, how="left").fillna(0.0)
    # FFT band powers
    fftf = _fft_band_powers_fast(X, value_col=value_col, Nf=Nf)
    feats = feats.join(fftf, how="left").fillna(0.0)

    # Safety
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats

# -------------------------------
# SAVE & PCA WRAPPERS
# -------------------------------
def build_and_save_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            Nf: int = 12,
                            out_dir: Path = RESOURCES_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    Z_train = f_x2z_expert_fast(X_train, Nf=Nf)
    Z_test  = f_x2z_expert_fast(X_test,  Nf=Nf)

    Z_train_path = out_dir / "Z_train.parquet"
    Z_test_path  = out_dir / "Z_test.parquet"
    Z_train.to_parquet(Z_train_path)
    Z_test.to_parquet(Z_test_path)
    print(f"[features] saved {Z_train.shape} -> {Z_train_path}")
    print(f"[features] saved {Z_test.shape}  -> {Z_test_path}")
    return Z_train, Z_test

def fit_pca_and_transform(X_tr: pd.DataFrame, X_te: pd.DataFrame,
                          n_components: int | float = 0.99,
                          out_dir: Path = RESOURCES_DIR):
    """
    Fit PCA on train (n_components as int or variance fraction), transform both,
    save PCA and transformed matrices.
    """
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    Ztr_pca = pca.fit_transform(X_tr.values)
    Zte_pca = pca.transform(X_te.values)

    Ztr_pca_df = pd.DataFrame(Ztr_pca, index=X_tr.index,
                              columns=[f"pc_{i}" for i in range(Ztr_pca.shape[1])])
    Zte_pca_df = pd.DataFrame(Zte_pca, index=X_te.index,
                              columns=[f"pc_{i}" for i in range(Zte_pca.shape[1])])

    joblib.dump(pca, out_dir / "pca.joblib")
    Ztr_pca_df.to_parquet(out_dir / "Z_train_pca.parquet")
    Zte_pca_df.to_parquet(out_dir / "Z_test_pca.parquet")
    print(f"[pca] kept {Ztr_pca.shape[1]} components; saved PCA + transformed matrices.")
    return Ztr_pca_df, Zte_pca_df