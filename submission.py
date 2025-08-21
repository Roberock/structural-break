# submission.py
# Deterministic, self-contained solution for the CrunchDAO Structural Break competition.
# - Variable-length series -> fixed-length features (robust stats + FFT + W1 via quantiles)
# - HistGradientBoostingClassifier as baseline model
# - Deterministic outputs (fixed seeds, no randomness at inference)
#
# Expected environment:
#   - Python 3.9+
#   - numpy, pandas, scikit-learn, joblib (scipy not strictly required here)
#
# NOTE: The platform will call train() and infer() as per the official template.

from __future__ import annotations

import os
import typing as t
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score


# ------------------------- Determinism helpers -------------------------

def _set_determinism(seed: int = 42) -> None:
    """
    Enforce deterministic behavior where possible.
    """
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass


# ------------------------- Feature Extraction -------------------------

_Q_LEVELS = np.linspace(0.0, 1.0, 101)  # for quantile-based Wasserstein


def _robust_stats(x: np.ndarray) -> dict[str, float]:
    """Robust summary stats for a 1D array."""
    if x.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "median": 0.0, "q10": 0.0, "q25": 0.0, "q75": 0.0, "q90": 0.0,
            "mad": 0.0, "skew": 0.0, "kurt": 0.0, "rms": 0.0, "ptp": 0.0,
        }
    q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
    med = q50
    # median absolute deviation
    mad = float(np.median(np.abs(x - med)))
    # simple skew/kurt without scipy
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-12
    z = (x - m) / s
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)  # Fisher's definition

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(med),
        "q10": float(q10), "q25": float(q25), "q75": float(q75), "q90": float(q90),
        "mad": float(mad),
        "skew": float(skew),
        "kurt": float(kurt),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "ptp": float(np.ptp(x)),
    }


def _fft_energy_bands(x: np.ndarray, n_bands: int = 5) -> dict[str, float]:
    """
    Very compact frequency summary: relative energy in n_bands of the (demeaned) power spectrum (rfft).
    """
    if x.size < 8:
        return {f"fft_band_{i}": 0.0 for i in range(n_bands)}
    spec = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    spec = spec[1:]  # drop DC for stability
    if spec.size == 0:
        return {f"fft_band_{i}": 0.0 for i in range(n_bands)}
    bands = np.array_split(spec, n_bands)
    energies = np.array([float(b.sum()) for b in bands], dtype=float)
    tot = float(energies.sum()) + 1e-12
    return {f"fft_band_{i}": float(e / tot) for i, e in enumerate(energies)}


def _w1_from_quantiles(pre: np.ndarray, post: np.ndarray, qs: np.ndarray = _Q_LEVELS) -> float:
    """
    Approximate Wasserstein-1 distance via quantiles on a uniform grid.
    W1(F,G) ≈ mean_u |Q_F(u) - Q_G(u)|.
    """
    if pre.size == 0 or post.size == 0:
        return 0.0
    q_pre = np.quantile(pre, qs, method="linear")
    q_post = np.quantile(post, qs, method="linear")
    return float(np.mean(np.abs(q_pre - q_post)))


@dataclass
class FeatureExtractor:
    """
    Maps a variable-length (value, period) sequence into a fixed-length feature vector.
    """
    n_fft_bands: int = 5

    # computed from first pass; used to reindex consistently
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _features_for_segment(self, x: np.ndarray, prefix: str) -> dict[str, float]:
        f = {}
        rs = _robust_stats(x)
        f.update({f"{prefix}_{k}": v for k, v in rs.items()})
        f.update({f"{prefix}_{k}": v for k, v in _fft_energy_bands(x, n_bands=self.n_fft_bands).items()})
        return f

    def _features_for_id(self, df_one: pd.DataFrame) -> dict[str, float]:
        # df_one: columns ["value", "period"], index: time
        vals = df_one["value"].to_numpy(dtype=float, copy=False)
        pre = df_one.loc[df_one["period"] == 0, "value"].to_numpy(dtype=float, copy=False)
        post = df_one.loc[df_one["period"] == 1, "value"].to_numpy(dtype=float, copy=False)

        d: dict[str, float] = {}

        # global + segment stats
        d.update(self._features_for_segment(vals, "g"))
        d.update(self._features_for_segment(pre, "pre"))
        d.update(self._features_for_segment(post, "post"))

        # deltas (post - pre) for key stats
        for k in ["mean", "std", "median", "mad", "skew", "kurt", "rms"]:
            d[f"delta_{k}"] = d.get(f"post_{k}", 0.0) - d.get(f"pre_{k}", 0.0)

        # probabilistic distance via quantiles (Wasserstein-1 approx)
        d["w1_quant"] = _w1_from_quantiles(pre, post)

        # lengths & ratio
        n_pre = float(pre.size)
        n_post = float(post.size)
        d["len_total"] = float(vals.size)
        d["n_pre"] = n_pre
        d["n_post"] = n_post
        d["ratio_post_pre"] = float(n_post / (n_pre + 1e-6))

        # Ensure finite
        for k, v in list(d.items()):
            if not np.isfinite(v):
                d[k] = 0.0

        return d

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for all ids in X and record feature names for consistent ordering.
        X has MultiIndex (id, time) and columns ["value", "period"].
        """
        out_rows = []
        id_list = []
        for id_, grp in X.groupby(level="id", sort=False):
            # drop the id level for convenience
            grp_id = grp.droplevel("id")
            feats = self._features_for_id(grp_id)
            out_rows.append(feats)
            id_list.append(id_)
        df = pd.DataFrame(out_rows, index=id_list).sort_index()
        # store canonical order
        self.feature_names_ = list(df.columns)
        # clean
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return df

    def transform_single(self, df_one_id: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for a single id (df for one time series), return a 1xD DataFrame
        with columns ordered as in feature_names_ (missing -> 0).
        """
        feats = self._features_for_id(df_one_id)
        row = pd.DataFrame([feats])
        if self.feature_names_:
            row = row.reindex(columns=self.feature_names_, fill_value=0.0)
        row = row.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        row.index = [0]  # single row
        return row


# ------------------------- Model bundle -------------------------

@dataclass
class ModelBundle:
    model: HistGradientBoostingClassifier
    feature_names: list[str]

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @staticmethod
    def load(path: str) -> "ModelBundle":
        obj = joblib.load(path)
        return ModelBundle(model=obj["model"], feature_names=obj["feature_names"])


# ------------------------- Training -------------------------

def _cv_tune_hgb(
    X: pd.DataFrame, y: pd.Series, seed: int = 42, enable_cv: bool = True
) -> HistGradientBoostingClassifier:
    """
    Small, deterministic grid search for ROC-AUC with 5-fold CV.
    If enable_cv=False, returns a strong default configuration.
    """
    if not enable_cv:
        return HistGradientBoostingClassifier(
            random_state=seed,
            early_stopping=True, validation_fraction=0.1,
            max_iter=600, learning_rate=0.06, max_depth=5,
            min_samples_leaf=25, l2_regularization=1e-2
        )

    base = HistGradientBoostingClassifier(
        random_state=seed,
        early_stopping=True, validation_fraction=0.1
    )
    param_grid = {
        "learning_rate": [0.03, 0.06],
        "max_depth": [5, 7],
        "max_iter": [400, 800],
        "min_samples_leaf": [25, 50],
        "l2_regularization": [1e-2, 1e-1],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,        # keep strict determinism
        verbose=0,
        refit=True
    )
    gs.fit(X, y)
    # optional log during training
    try:
        print(f"[CV] best ROC-AUC: {gs.best_score_:.6f} with {gs.best_params_}")
    except Exception:
        pass
    return gs.best_estimator_


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    """
    Train a deterministic classifier on extracted features and save bundle to model_directory_path.
    - X_train: MultiIndex (id, time), columns ["value", "period"]
    - y_train: pd.Series indexed by id with values {0,1} or {False,True}
    """
    _set_determinism(42)

    # Ensure y is Series of ints indexed by id
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    y_train = y_train.astype(int).copy()

    # Feature extraction
    fe = FeatureExtractor(n_fft_bands=5)
    X_feats = fe.fit_transform(X_train)  # records feature_names_
    # align target
    y_aligned = y_train.loc[X_feats.index]

    # Optional CV (off by default for speed—set env ENABLE_CV=1 to enable)
    enable_cv = os.environ.get("ENABLE_CV", "0") == "1"
    clf = _cv_tune_hgb(X_feats, y_aligned, seed=42, enable_cv=enable_cv)

    # Fit on all training features
    clf.fit(X_feats, y_aligned)

    # (optional) quick internal AUC print if you pass a validation split separately
    try:
        # trivial sanity: train AUC (not a validation metric)
        y_scores = clf.predict_proba(X_feats)[:, 1]
        auc = roc_auc_score(y_aligned, y_scores)
        print(f"[train] Train ROC-AUC (sanity): {auc:.6f}")
    except Exception:
        pass

    # Persist bundle
    os.makedirs(model_directory_path, exist_ok=True)
    bundle = ModelBundle(model=clf, feature_names=fe.feature_names_)
    bundle_path = os.path.join(model_directory_path, "model.joblib")
    bundle.save(bundle_path)
    print(f"[train] Saved model bundle -> {bundle_path}")


# ------------------------- Inference -------------------------

def infer(
    X_test: t.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    """
    Generator expected by the platform:
      1) load model,
      2) yield once to signal readiness,
      3) for each dataset (one id), compute P(y=1) and yield a scalar in [0,1].

    Each `dataset` is a DataFrame with MultiIndex (id, time) or a single-id
    frame with index 'time' if the runner already sliced by id.
    """
    _set_determinism(42)

    bundle_path = os.path.join(model_directory_path, "model.joblib")
    bundle = ModelBundle.load(bundle_path)
    clf = bundle.model
    feature_names = bundle.feature_names

    # Create a fresh extractor with saved schema
    fe = FeatureExtractor(n_fft_bands=5)
    fe.feature_names_ = feature_names

    # Mark as ready
    yield

    # Iterate ONCE over the provided datasets
    for dataset in X_test:
        # Handle both possible index layouts
        if isinstance(dataset.index, pd.MultiIndex) and "id" in dataset.index.names:
            # assume single id per dataset
            unique_ids = dataset.index.get_level_values("id").unique()
            if len(unique_ids) != 1:
                # If multiple ids are ever batched, process one-by-one in a stable order
                preds = []
                for _id in unique_ids:
                    df_one = dataset.xs(_id, level="id")
                    x_row = fe.transform_single(df_one)
                    p = float(clf.predict_proba(x_row)[:, 1][0])
                    preds.append((int(_id), p))
                # If multiple, return a mean or raise — but spec suggests single-id datasets.
                # Here we consistently average to yield a single scalar.
                p_out = float(np.mean([p for _, p in preds])) if preds else 0.0
            else:
                df_one = dataset.droplevel("id")
                x_row = fe.transform_single(df_one)
                p_out = float(clf.predict_proba(x_row)[:, 1][0])
        else:
            # index is just time, single id
            x_row = fe.transform_single(dataset)
            p_out = float(clf.predict_proba(x_row)[:, 1][0])

        # yield a scalar score in [0,1] = P(y=1 | X, p)
        yield p_out
