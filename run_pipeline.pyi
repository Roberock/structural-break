import numpy as np
import pandas as pd
from pathlib import Path

# --- Rich features: robust stats + higher moments + FFT + wavelets ---
from scipy.stats import kurtosis, skew
from numpy.fft import rfft
import pywt  # pip install PyWavelets
from scipy.stats import wasserstein_distance, ks_2samp

# --- train model ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from preprocess import load_data

RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)


## 1. START WITH FEATURES EXTRACTION
def _stats_robust(x):
    if x.size == 0:
        return {}
    q5, q10, q25, q50, q75, q90, q95= np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    mad = np.median(np.abs(x - q50))  # Median Absolute Deviation
    return {
        "mean": np.mean(x), "std": np.std(x), "min": np.min(x),  "max": np.max(x),
        "median": q50,  "q5": q5, "q10": q10, "q25": q25, "q75": q75, "q90": q90,  "q95": q5,
        "mad": mad, "skew": skew(x, bias=False) if x.size > 2 else 0.0, "kurt": kurtosis(x, fisher=True, bias=False) if x.size > 3 else 0.0,
        "rms": np.sqrt(np.mean(x**2)),  "ptp": np.ptp(x),  # peak-to-peak
    }

def _fft_energy_bands(x, n_bands=5):
    """Simple relative-band energies from power spectrum."""
    if x.size < 8:
        return {f"fft_band_{i}": 0.0 for i in range(n_bands)}
    spec = np.abs(rfft(x - np.mean(x)))**2
    spec = spec[1:]  # drop DC for energy ratios
    if spec.size == 0:
        return {f"fft_band_{i}": 0.0 for i in range(n_bands)}
    # split into equal bands
    bands = np.array_split(spec, n_bands)
    energies = np.array([b.sum() for b in bands], dtype=float)
    tot = energies.sum() + 1e-12
    return {f"fft_band_{i}": float(e/tot) for i, e in enumerate(energies)}

def _wavelet_energies(x, wavelet="db4", level=None):
    """Wavelet packet-ish: energy per scale from DWT coefficients."""
    if x.size < 8:
        return {}
    coeffs = pywt.wavedec(x - np.mean(x), wavelet=wavelet, level=level)
    energies = [np.sum(c**2) for c in coeffs]  # [cA_L, cD_L, ..., cD1]
    tot = np.sum(energies) + 1e-12
    out = {"wl_cA": float(energies[0]/tot)}
    for i, e in enumerate(energies[1:], start=1):
        out[f"wl_cD_{i}"] = float(e/tot)
    return out

def _segment_features(x):
    """Compose robust stats + spectral features for one segment."""
    f = {}
    f.update(_stats_robust(x))
    f.update(_fft_energy_bands(x, n_bands=5))
    f.update(_wavelet_energies(x))
    return f

def extract_features_rich(X: pd.DataFrame) -> pd.DataFrame:
    """Per-id features using value and period columns."""
    feats = []
    for id_, g in X.groupby(level="id"):
        v = g["value"].values.astype(float)
        pre = g.loc[g["period"] == 0, "value"].values.astype(float)
        post = g.loc[g["period"] == 1, "value"].values.astype(float)

        d = {"id": id_}
        # global signal fZ(xt)
        d.update({f"g_{k}": v for k, v in _segment_features(v).items()})

        # pre/post (fZ(xt[period==1]) and fZ(xt[period=0]))
        d.update({f"pre_{k}": v for k, v in _segment_features(pre).items()})
        d.update({f"post_{k}": v for k, v in _segment_features(post).items()})

        # deltas (post - pre) for key stats
        for k in ["mean", "std", "median", "mad", "skew", "kurt", "rms"]:
            d[f"delta_{k}"] = (d.get(f"post_{k}", 0.0) - d.get(f"pre_{k}", 0.0))

        # counts & ratio
        d["len_total"] = int(v.size)
        d["n_pre"] = int(pre.size)
        d["n_post"] = int(post.size)
        d["ratio_post_pre"] = float(d["n_post"]/(d["n_pre"]+1e-6))
        feats.append(d)

    df = pd.DataFrame(feats).set_index("id")
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


## 2. CV training of HistGradientBoostingClassifier
def train_cv_tuned_model(X: pd.DataFrame, y: pd.Series):
    """
    Tune HGB hyperparameters with StratifiedKFold on ROC-AUC, then refit on all data.
    Returns (best_estimator, best_params, cv_results_df).
    """
    est = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True,          # quick regularisation
        validation_fraction=0.1,
    )

    # A compact grid for ROC-AUC .... expand if time allows
    param_grid = {
        "learning_rate": [0.03, 0.06, 0.1],
        "max_depth": [3, 5, 7],
        "max_iter": [200, 400, 800],
        "min_samples_leaf": [10, 25, 50],
        "l2_regularization": [0.0, 1e-2, 1e-1],
        "max_bins": [255],            # default is strong; tune if you wish
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sample_weight = compute_sample_weight("balanced", y) # handle possible class imbalance

    gs = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,   # refit on full data with best params automatically
    )

    gs.fit(X, y, sample_weight=sample_weight)
    best_est = gs.best_estimator_
    best_params = gs.best_params_
    cv_results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")

    print("\n[CV] Best ROC-AUC (mean cv):", gs.best_score_)
    print("[CV] Best params:", best_params)
    return best_est, best_params, cv_results


def evaluate_on_holdout(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Report holdout ROC-AUC and return probability DataFrame [P(y=0), P(y=1)]."""
    proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba[:, 1])
    print("[Holdout] ROC-AUC:", auc)
    proba_df = pd.DataFrame(
        proba, index=X_test.index, columns=["P(y=0)", "P(y=1)"]
    )
    return auc, proba_df


if __name__ == "__main__":
    # ==== end-to-end ====
    # 0) Load data
    X_raw, X_test_raw, y_train, y_test = load_data('../data')

    # 1) extract features
    FEATURE_DIR = Path("../data/feature/extraction")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    X_train_moments = extract_features_rich(X_raw).fillna(0.0)
    X_test_moments = extract_features_rich(X_test_raw).fillna(0.0)

    X_train_moments.to_parquet(FEATURE_DIR / "X_train_moments.parquet")
    X_test_moments.to_parquet(FEATURE_DIR / "X_test_moments.parquet")

    # 2) CV + tuning, refit on all training
    model, best_params, cv_results = train_cv_tuned_model(X_train_moments, y_train)

    # 3) Evaluate on your provided X_test / y_test
    holdout_auc, proba_test = evaluate_on_holdout(model, X_test_moments, y_test)

    # 4) Persist artifacts
    joblib.dump(model, RESOURCES_DIR / "hgb_structbreak_model.joblib")
    cv_results.to_csv(RESOURCES_DIR / "cv_results.csv", index=False)
    proba_test.to_parquet(RESOURCES_DIR / "proba_test.parquet")

    print("\nSaved:")
    print("-", RESOURCES_DIR / "hgb_structbreak_model.joblib")
    print("-", RESOURCES_DIR / "cv_results.csv")
    print("-", RESOURCES_DIR / "proba_test.parquet")