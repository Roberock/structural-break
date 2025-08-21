
from preprocess import train_vae_on_X, vae_embeddings

def load_data():
    """Load parquet training data."""
    DATA_DIR = Path("data")
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze()
    return X_train, y_train


def train_model(X_tr, y_tr):
    """Train a baseline model."""
    # 1) Train VAE unsupervised on X (any feature set you choose)
    FEATURE_COLS = list(X_tr.columns)  # works for any dimension
    vae = train_vae_on_X(X_tr, feature_cols=FEATURE_COLS, latent_dim=64, epochs=10, beta=1.0)

    # 2) Extract id-level embeddings μ (fixed-length features)
    X_emb = vae_embeddings(vae, X_tr, feature_cols=FEATURE_COLS)  # index = id

    # 3) Train a classifier to maximise ROC-AUC
    clf = LogisticRegression(max_iter=10_000, class_weight="balanced")  # start simple
    clf.fit(X_emb.loc[y_tr.index], y_tr.values)
    scores = clf.predict_proba(X_emb)[:, 1]
    print("Train ROC-AUC:", roc_auc_score(y_tr.loc[X_emb.index], scores))

    # 4) Persist
    joblib.dump(vae.state_dict(), "resources/vae_state_dict.pt")
    joblib.dump(clf, "resources/vae_logreg.joblib")

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    return model


def save_model(model):
    """Save the trained model to resources/."""
    model_path = RESOURCES_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"[train_model] Model saved to {model_path}")

