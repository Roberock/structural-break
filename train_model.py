from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt
from extract_features import *
from preprocess import X_to_wide

RESOURCES_DIR = Path("resources")
RESOURCES_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":

        # import data

        from preprocess import load_data
        X_train, X_test, y_train, y_test = load_data()

        # EXAMPLE OF PLOT
        # pick 2 ids for each class
        ids_true = y_train[y_train == True].sample(2, random_state=42).index
        ids_false = y_train[y_train == False].sample(2, random_state=42).index
        selected_ids = list(ids_true) + list(ids_false)

        print("Selected ids:", selected_ids)

        # plot example
        fig, axes = plt.subplots(len(selected_ids), 1, figsize=(10, 8), sharex=True)

        for ax, i in zip(axes, selected_ids):
            seq = X_train.loc[i]
            ax.plot(seq.index, seq["value"], label=f"value, id={i}, y={y_train.loc[i]}")
            seq = X_train.loc[i]
            ax.plot(seq.index, seq["period"]/50, label=f"period, id={i}, y={y_train.loc[i]}")
            ax.set_ylabel("X")
            ax.legend()
            ax.grid(True)

        plt.xlabel("time")
        plt.suptitle("Example sequences from X_train")
        plt.tight_layout()
        plt.show()

        # map to big column
        X_train_wide = X_to_wide(X_train)

        # train model
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        # build wide train + align target
        X_train_wide = X_train_wide.loc[y_train.index]  # ensure same ids

        # fill NaNs with 0 (or could use mean imputation)
        X_train_wide = X_train_wide.fillna(0.0)

        # quick train/valid split (if you don't want to use X_test yet)
        Xtr, Xval, ytr, yval = train_test_split(
            X_train_wide, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        clf = LogisticRegression(
            max_iter=10000,
            solver="saga",  # handles L1/L2 on large sparse-ish data
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            n_jobs=-1
        )

        clf.fit(Xtr, ytr)

        # validation ROC-AUC
        p_val = clf.predict_proba(Xval)[:, 1]
        print("Validation ROC-AUC:", roc_auc_score(yval, p_val))

        # ===== apply to test set =====
        X_test_wide = X_to_wide(X_test).fillna(0.0)
        X_test_wide = X_test_wide.loc[y_test.index]  # align
        p_test = clf.predict_proba(X_test_wide)[:, 1]

        print("Test ROC-AUC:", roc_auc_score(y_test, p_test))

        # feature extraction here
        Z_train = f_x2z_expert(X_train)
        Z_test = f_x2z_expert(X_test)


        # after you loaded X_train, X_test
        # Z_train, Z_test = build_and_save_features(X_train, X_test, Nf=12)

        # OPTIONAL: PCA to denoise / compress (keeps 99% variance)
        Z_train_pca, Z_test_pca = fit_pca_and_transform(X_train, X_test, n_components=0.95)

        from pathlib import Path
        import numpy as np
        import pandas as pd
        import joblib

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        from sklearn.metrics import roc_auc_score

        RESOURCES_DIR = Path("resources")
        RESOURCES_DIR.mkdir(exist_ok=True)


        def train_predict_Z_to_Y(
                Z_train: pd.DataFrame,
                y_train: pd.Series,
                Z_test: pd.DataFrame,
                y_test: pd.Series | None = None,
                cv_splits: int = 5,
                random_state: int = 42,
        ):
            """
            Train a classifier to map Z -> Y with AUC-centric CV model selection.
            Returns: dict with model, cv_results, preds_test (Series of probabilities).
            Saves: resources/model.joblib, resources/preds_test.csv (and preds_train.csv).
            """

            # Ensure alignment and numpy arrays
            Ztr = Z_train.loc[y_train.index]
            ytr = y_train.astype(int).values
            Zte = Z_test.copy()

            # ====== Models & param grids ======
            models = [
                (
                    "logreg",
                    Pipeline([
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("clf", LogisticRegression(
                            max_iter=10000, class_weight="balanced", solver="lbfgs"))
                    ]),
                    {
                        "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0]
                    }
                ),
                (
                    "gbrt",
                    GradientBoostingClassifier(random_state=random_state),
                    {
                        "n_estimators": [200, 400],
                        "max_depth": [2, 3],
                        "learning_rate": [0.03, 0.06, 0.1],
                        "subsample": [0.8, 1.0]
                    }
                ),
            ]

            skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

            best = {"name": None, "estimator": None, "score": -np.inf, "cv": None}

            for name, est, grid in models:
                gs = GridSearchCV(
                    est, grid,
                    scoring="roc_auc",
                    cv=skf,
                    n_jobs=-1,
                    refit=True,
                    verbose=0
                )
                gs.fit(Ztr.values, ytr)
                mean_auc = gs.best_score_
                print(f"[CV] {name}: best AUC={mean_auc:.5f}  params={gs.best_params_}")
                if mean_auc > best["score"]:
                    best.update(
                        {"name": name, "estimator": gs.best_estimator_, "score": mean_auc, "cv": gs.cv_results_})

            model = best["estimator"]
            print(f"[select] Best model: {best['name']} (CV AUC={best['score']:.5f})")

            # Fit on all training data (GridSearch already refit=True, but re-fit for clarity)
            model.fit(Ztr.values, ytr)

            # Predictions
            p_tr = model.predict_proba(Ztr.values)[:, 1]
            preds_train = pd.Series(p_tr, index=Ztr.index, name="proba")

            p_te = model.predict_proba(Zte.values)[:, 1]
            preds_test = pd.Series(p_te, index=Zte.index, name="proba")

            # Optional test evaluation
            if y_test is not None:
                # align test labels to Z_test index
                yte = y_test.loc[Zte.index].astype(int).values
                test_auc = roc_auc_score(yte, p_te)
                print(f"[holdout] Test ROC-AUC: {test_auc:.5f}")

            # Save artifacts
            joblib.dump(model, RESOURCES_DIR / "model.joblib")
            preds_test.to_csv(RESOURCES_DIR / "preds_test.csv", header=True)
            preds_train.to_csv(RESOURCES_DIR / "preds_train.csv", header=True)
            print("[save] model -> resources/model.joblib")
            print("[save] preds -> resources/preds_train.csv, resources/preds_test.csv")

            return {
                "model": model,
                "best_name": best["name"],
                "cv_auc": best["score"],
                "cv_results": best["cv"],
                "preds_test": preds_test,
                "preds_train": preds_train,
            }
