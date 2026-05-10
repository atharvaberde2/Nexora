"""Nexora Flask backend.

Endpoints:
    POST /api/csv/inspect         pandas-authoritative column types
    POST /api/audit/stage/1       Stage 1 — Bias Fingerprinting (multi-protected)
    POST /api/audit/stage/2       Stage 2 — Multi-model training via Optuna

Run:
    python backend/app.py
"""

import json
import math
import time
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import numpy as np
import optuna
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from scipy.stats import chi2, chi2_contingency, norm
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Optional boosted-tree libraries — fall back to sklearn HistGradientBoosting
# if the user hasn't installed them.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore[assignment]
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    LGBMClassifier = None  # type: ignore[assignment]
    HAS_LGBM = False

try:
    import shap as shap_lib
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Quiet down Optuna's per-trial chatter — the Flask logs stay readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_TRIALS = 10
CV_FOLDS = 5
SCORING = "roc_auc"
BOOTSTRAP_N = 1000
N_THRESHOLD = 100  # Power threshold for stage 1 + small-group warning in 3

# Stage 4 selection rule
EO_GAP_THRESHOLD = 0.15   # Equalized-odds gap above this disqualifies a model
                          # from being "recommended" even if AUC is high.
COMPOSITE_LAMBDA = 1.5    # Penalty weight for EO gap in optional composite score
                          # (Score = AUC − λ·EO_gap).

# Stage 5
SHAP_EXPLAIN_N = 500   # max samples passed to SHAP explainer
SHAP_BG_N = 100        # background samples for LinearExplainer
PERM_N = 200           # permutation test shuffles
S5_TOP_N = 10          # top features to return

# Session cache: maps session_id (issued during Stage 2) → SessionState dict.
# Keeps only the latest few audits in-memory; eviction prevents unbounded growth.
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
SESSION_LIMIT = 8


def _store_session(session_id: str, payload: dict) -> None:
    SESSIONS[session_id] = payload
    SESSIONS.move_to_end(session_id)
    while len(SESSIONS) > SESSION_LIMIT:
        SESSIONS.popitem(last=False)


def _get_session(session_id: str) -> dict | None:
    s = SESSIONS.get(session_id)
    if s is not None:
        SESSIONS.move_to_end(session_id)
    return s


app = Flask(__name__)
CORS(app)


# =========================================================
# Helpers
# =========================================================

def _read_csv_from_request() -> pd.DataFrame:
    upload = request.files["file"]
    text = upload.read().decode("utf-8", errors="replace")
    if text.startswith("﻿"):
        text = text[1:]
    df = pd.read_csv(StringIO(text))
    df.columns = df.columns.str.strip()
    return df


def _safe_float(x) -> float | None:
    """Convert to float, mapping NaN/Inf/None → None for clean JSON."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# =========================================================
# Column inspection (Pandas-authoritative types)
# =========================================================

NA_LIKE = {"", "na", "n/a", "null", "nan", "none"}


def _inspect_column(s: pd.Series) -> dict:
    n = len(s)
    n_missing_native = int(s.isnull().sum())

    # Normalize values for unique-counting: drop NaN and string NA-likes.
    if s.dtype == object:
        cleaned = s.dropna().astype(str).str.strip()
        cleaned = cleaned[~cleaned.str.lower().isin(NA_LIKE)]
    else:
        cleaned = s.dropna()

    # Effective missing includes string NA-likes
    n_missing = n - len(cleaned)
    unique_vals = cleaned.unique().tolist()
    unique_count = len(unique_vals)

    col_type: str
    numeric_range: list[float] | None = None

    # Precedence: binary > numeric > categorical > freetext.
    # binary takes priority so 0/1 encoded sex is correctly typed.
    if unique_count == 2:
        col_type = "binary"
    elif pd.api.types.is_numeric_dtype(s):
        col_type = "numeric"
        try:
            numeric_range = [float(cleaned.min()), float(cleaned.max())]
        except (TypeError, ValueError):
            numeric_range = None
    else:
        coerced = pd.to_numeric(cleaned, errors="coerce")
        ok_ratio = coerced.notnull().sum() / max(len(cleaned), 1)
        if ok_ratio > 0.95:
            col_type = "numeric"
            try:
                numeric_range = [float(coerced.min()), float(coerced.max())]
            except (TypeError, ValueError):
                numeric_range = None
        elif unique_count <= 30:
            col_type = "categorical"
        else:
            col_type = "freetext"

    sample = [str(v) for v in unique_vals[:6]]
    unique_values = (
        [str(v) for v in unique_vals[:30]]
        if col_type in ("binary", "categorical")
        else None
    )

    return {
        "name": str(s.name),
        "type": col_type,
        "unique": unique_count,
        "missing_pct": (n_missing / n) if n > 0 else 0.0,
        "sample": sample,
        "unique_values": unique_values,
        "numeric_range": numeric_range,
    }


# =========================================================
# STAGE 0 — preprocessing
# =========================================================

def preprocess(df: pd.DataFrame, target_col: str, protected_col: str, positive_class: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not found")
    if protected_col not in df.columns:
        raise KeyError(f"protected column '{protected_col}' not found")

    # Protected attribute — strip and normalize blanks/NaN-likes to NaN
    df[protected_col] = (
        df[protected_col].astype(str).str.strip()
        .replace(["nan", "NaN", "None", "", " "], np.nan)
    )

    # Target — encode 1 if equals the user's chosen positive class, else 0.
    # Respects the UI's positive-class selection rather than relying on
    # alphabetical LabelEncoder ordering.
    pos = str(positive_class).strip()
    df[target_col] = (df[target_col].astype(str).str.strip() == pos).astype(int)

    return df


# =========================================================
# STAGE 1 — bias fingerprinting
# =========================================================

def bias_fingerprint(df: pd.DataFrame, target_col: str, protected_col: str) -> dict:
    df = df.dropna(subset=[target_col, protected_col]).copy()

    if len(df) == 0:
        raise ValueError("no rows remaining after dropping NaN target/protected values")

    group_pos_rate = df.groupby(protected_col)[target_col].mean()
    group_missing = (
        df.isnull().groupby(df[protected_col]).mean().mean(axis=1)
    )
    overall_rate = float(df[target_col].mean())
    group_base_gap = group_pos_rate - overall_rate
    counts = df[protected_col].value_counts()

    contingency = pd.crosstab(df[protected_col], df[target_col])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        chi2_stat, p_value, significant = None, None, False
    else:
        try:
            chi2_stat, p_value, _, _ = chi2_contingency(contingency)
            chi2_stat = _safe_float(chi2_stat)
            p_value = _safe_float(p_value)
            significant = bool(p_value is not None and p_value < 0.05)
        except Exception:
            chi2_stat, p_value, significant = None, None, False

    col_missing = df.isnull().mean().sort_values(ascending=False).head(4)

    n_threshold = 100
    groups = []
    for g_name, n in counts.items():
        if pd.isna(g_name):
            continue
        n_int = int(n)
        groups.append({
            "name": str(g_name),
            "n": n_int,
            "positive_rate": _safe_float(group_pos_rate.get(g_name)),
            "base_rate_gap": _safe_float(group_base_gap.get(g_name)),
            "missing_rate": _safe_float(group_missing.get(g_name)),
            "sufficient_power": n_int >= n_threshold,
        })
    groups.sort(key=lambda g: -g["n"])

    return {
        "n_total": int(len(df)),
        "overall_positive_rate": _safe_float(overall_rate),
        "n_threshold": n_threshold,
        "groups": groups,
        "label_bias": {
            "chi2": chi2_stat,
            "p_value": p_value,
            "significant": significant,
        },
        "top_missing_columns": [
            {"name": str(name), "missing_pct": _safe_float(pct)}
            for name, pct in col_missing.items()
        ],
    }


# =========================================================
# STAGE 2 — multi-model training via Optuna
# =========================================================

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    protected_cols: list[str],
    positive_class: str,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray], list[str]]:
    """Encode target, drop target+protected from features, impute, one-hot.

    Returns (X, y, n_features, protected_values, feature_names).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not found")

    pos = str(positive_class).strip()
    y_series = (df[target_col].astype(str).str.strip() == pos).astype(int)

    drop_cols = {target_col, *protected_cols}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Drop rows whose target is missing
    keep_mask = df[target_col].notna()
    X = X[keep_mask]
    y = y_series[keep_mask].values

    # Capture protected values aligned with the kept rows (before any further
    # X transformations — protected lives only in `df`, never in X)
    protected_values: dict[str, np.ndarray] = {}
    for col in protected_cols:
        if col in df.columns:
            protected_values[col] = (
                df[col][keep_mask].astype(str).str.strip().values
            )

    # Drop high-cardinality non-numeric columns (likely IDs / freetext)
    for c in list(X.columns):
        if X[c].dtype == object and X[c].nunique(dropna=True) > 30:
            X = X.drop(columns=[c])

    # Coerce object-encoded numerics where >95% parses (e.g. "12.3" strings)
    for c in list(X.columns):
        if X[c].dtype == object:
            coerced = pd.to_numeric(X[c], errors="coerce")
            if coerced.notna().sum() / max(len(coerced), 1) > 0.95:
                X[c] = coerced

    # Impute: median for numeric, "MISSING" for object
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        med = X[c].median()
        if pd.isna(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    obj_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].fillna("MISSING").astype(str)
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=False)

    feature_names = X.columns.tolist()
    return X.values.astype(float), y.astype(int), X.shape[1], protected_values, feature_names


def _cv(X: np.ndarray, y: np.ndarray, model) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=SCORING)
    return float(np.mean(scores)), float(np.std(scores))


def _objective_logreg(trial, X, y):
    C = trial.suggest_float("C", 1e-4, 1e2, log=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000),
    )
    return _cv(X, y, model)[0]


def _objective_rf(trial, X, y):
    # n_jobs=1 because the 5 models run in parallel via ThreadPoolExecutor —
    # using -1 here would oversubscribe the CPU.
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 150),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=42,
        n_jobs=1,
    )
    return _cv(X, y, model)[0]


def _objective_xgb(trial, X, y):
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            max_iter=trial.suggest_int("max_iter", 50, 200),
            random_state=42,
        )
    return _cv(X, y, model)[0]


def _objective_lgbm(trial, X, y):
    if HAS_LGBM:
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            max_iter=trial.suggest_int("max_iter", 50, 200),
            random_state=42,
        )
    return _cv(X, y, model)[0]


def _objective_svm(trial, X, y):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=C, dual="auto", max_iter=5000),
    )
    return _cv(X, y, model)[0]


def _build_model_from_params(key: str, params: dict):
    """Re-instantiate the best estimator so we can compute final cv_mean/std."""
    if key == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=params["C"], penalty="l2", solver="lbfgs", max_iter=1000
            ),
        )
    if key == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=1, **params)
    if key == "xgb":
        if HAS_XGB:
            return XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
                **params,
            )
        return HistGradientBoostingClassifier(random_state=42, **params)
    if key == "lgbm":
        if HAS_LGBM:
            return LGBMClassifier(random_state=42, verbose=-1, n_jobs=1, **params)
        return HistGradientBoostingClassifier(random_state=42, **params)
    if key == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=params["C"], dual="auto", max_iter=5000),
        )
    raise ValueError(f"unknown model key: {key}")


MODEL_REGISTRY = [
    {
        "key": "logistic",
        "name": "Logistic Regression",
        "family": "Linear · interpretable",
        "color": "m1",
        "objective": _objective_logreg,
        "available": True,
        "fallback_note": None,
    },
    {
        "key": "rf",
        "name": "Random Forest",
        "family": "Ensemble · bagged trees",
        "color": "m2",
        "objective": _objective_rf,
        "available": True,
        "fallback_note": None,
    },
    {
        "key": "xgb",
        "name": "XGBoost" if HAS_XGB else "HistGradientBoosting (xgb fallback)",
        "family": "Gradient boosting",
        "color": "m3",
        "objective": _objective_xgb,
        "available": HAS_XGB,
        "fallback_note": None if HAS_XGB else "xgboost not installed; using sklearn HistGradientBoosting",
    },
    {
        "key": "lgbm",
        "name": "LightGBM" if HAS_LGBM else "HistGradientBoosting (lgbm fallback)",
        "family": "Gradient boosting",
        "color": "m4",
        "objective": _objective_lgbm,
        "available": HAS_LGBM,
        "fallback_note": None if HAS_LGBM else "lightgbm not installed; using sklearn HistGradientBoosting",
    },
    {
        "key": "svm",
        "name": "Linear SVM",
        "family": "Max-margin classifier",
        "color": "m5",
        "objective": _objective_svm,
        "available": True,
        "fallback_note": None,
    },
]


def _cv_predictions(model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions via StratifiedKFold. Falls back to
    decision_function for estimators that don't expose predict_proba
    (e.g. LinearSVC) — we min/max-normalize the scores so AUC ranking
    still works, even though they're not calibrated probabilities."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf, method="predict")
    try:
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
        y_proba = proba[:, 1]
    except (AttributeError, ValueError):
        decision = cross_val_predict(model, X, y, cv=skf, method="decision_function")
        d_min, d_max = float(decision.min()), float(decision.max())
        if d_max - d_min < 1e-9:
            y_proba = np.zeros_like(decision, dtype=float)
        else:
            y_proba = (decision - d_min) / (d_max - d_min)
    return y_pred.astype(int), y_proba.astype(float)


def _train_one(spec: dict, X: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Train one model with Optuna + CV. Always returns
    (result_dict, y_pred, y_proba). On error y_pred/y_proba are None."""
    t0 = time.perf_counter()
    base = {
        "key": spec["key"],
        "name": spec["name"],
        "family": spec["family"],
        "color": spec["color"],
        "available": spec["available"],
        "fallback_note": spec["fallback_note"],
        "n_trials": 0,
        "best_score": None,
        "best_params": {},
        "cv_mean": None,
        "cv_std": None,
        "train_time_sec": None,
        "status": "error",
        "error": None,
    }
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial, sp=spec: sp["objective"](trial, X, y),
            n_trials=N_TRIALS,
            show_progress_bar=False,
        )
        best_params = dict(study.best_params)
        final_model = _build_model_from_params(spec["key"], best_params)
        cv_mean, cv_std = _cv(X, y, final_model)
        # Out-of-fold predictions used by Stages 3/4 — every row is predicted
        # by a fold-fitted estimator that didn't see it during training.
        y_pred, y_proba = _cv_predictions(final_model, X, y)
        base.update({
            "n_trials": N_TRIALS,
            "best_score": _safe_float(study.best_value),
            "best_params": {k: _json_safe(v) for k, v in best_params.items()},
            "cv_mean": _safe_float(cv_mean),
            "cv_std": _safe_float(cv_std),
            "train_time_sec": _safe_float(time.perf_counter() - t0),
            "status": "done",
            "error": None,
        })
        return base, y_pred, y_proba
    except Exception as e:
        base.update({
            "train_time_sec": _safe_float(time.perf_counter() - t0),
            "status": "error",
            "error": str(e),
        })
        return base, None, None


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _safe_float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


# =========================================================
# STAGE 3 — per-model fairness audit
# =========================================================

EPS = 1e-9


def _confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Returns (TPR, FPR, FNR, TNR, selection_rate, PPV)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    tnr = tn / (tn + fp + EPS)
    selection_rate = (tp + fp) / max(len(y_true), 1)
    ppv = tp / (tp + fp + EPS)   # Precision / Predictive Positive Value
    return tpr, fpr, fnr, tnr, selection_rate, ppv


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float | None:
    """Expected Calibration Error via equal-width probability bins."""
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_proba[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return _safe_float(ece)


def _delong_pvalue(y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray) -> float | None:
    """DeLong test for H0: AUC(a) == AUC(b) for paired classifiers on the same dataset."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos < 3 or n_neg < 3:
        return None
    try:
        def _placements(proba: np.ndarray):
            pos_p = proba[pos_idx]
            neg_p = proba[neg_idx]
            v10 = np.array(
                [np.mean(p > neg_p) + 0.5 * np.mean(p == neg_p) for p in pos_p]
            )
            v01 = np.array(
                [np.mean(pos_p > n) + 0.5 * np.mean(pos_p == n) for n in neg_p]
            )
            return v10, v01

        v10_a, v01_a = _placements(proba_a)
        v10_b, v01_b = _placements(proba_b)

        s10 = np.cov(np.vstack([v10_a, v10_b])) / n_pos
        s01 = np.cov(np.vstack([v01_a, v01_b])) / n_neg
        s = s10 + s01
        var_diff = s[0, 0] + s[1, 1] - 2.0 * s[0, 1]
        if var_diff <= 1e-14:
            return None
        z = (v10_a.mean() - v10_b.mean()) / np.sqrt(var_diff)
        return _safe_float(float(2.0 * norm.sf(abs(z))))
    except Exception:
        return None


def _mcnemar_pvalue(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float | None:
    """McNemar's test for H0: both models make the same errors (chi² with continuity correction)."""
    err_a = y_pred_a != y_true
    err_b = y_pred_b != y_true
    b = int((err_a & ~err_b).sum())   # A wrong, B right
    c = int((~err_a & err_b).sum())   # A right, B wrong
    if b + c < 5:
        return None
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return _safe_float(float(chi2.sf(stat, df=1)))


def _bh_fdr(pvalues: list) -> list:
    """Benjamini-Hochberg FDR correction. None entries pass through unchanged."""
    valid_pairs = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    if not valid_pairs:
        return list(pvalues)
    m = len(valid_pairs)
    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
    result = list(pvalues)
    prev = 1.0
    for rank, (i, p) in reversed(list(enumerate(sorted_pairs, 1))):
        adj = min(prev, p * m / rank)
        result[i] = _safe_float(min(adj, 1.0))
        prev = adj
    return result


def _percentile_ci(samples: list[float], alpha: float = 0.05) -> tuple[float | None, float | None]:
    if len(samples) < 2:
        return None, None
    arr = np.asarray(samples, dtype=float)
    lo, hi = np.percentile(arr, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return _safe_float(lo), _safe_float(hi)


def _bootstrap_auc_ci(y_true: np.ndarray, y_proba: np.ndarray, n: int, seed: int) -> tuple[float | None, float | None]:
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    samples: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, size=n_rows)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(float(roc_auc_score(y_true[idx], y_proba[idx])))
        except Exception:
            continue
    return _percentile_ci(samples)


def _group_metrics(
    y_true_g: np.ndarray, y_pred_g: np.ndarray, y_proba_g: np.ndarray, seed: int
) -> dict:
    n = len(y_true_g)
    if n == 0:
        return {
            "n": 0,
            "tpr": None, "tpr_ci": [None, None],
            "fpr": None, "fpr_ci": [None, None],
            "fnr": None, "tnr": None,
            "ppv": None, "ppv_ci": [None, None],
            "selection_rate": None,
            "auc": None, "auc_ci": [None, None],
            "sufficient_power": False,
        }

    tpr, fpr, fnr, tnr, sr, ppv = _confusion_rates(y_true_g, y_pred_g)
    try:
        auc = float(roc_auc_score(y_true_g, y_proba_g)) if len(np.unique(y_true_g)) >= 2 else None
    except Exception:
        auc = None

    tpr_samples: list[float] = []
    fpr_samples: list[float] = []
    ppv_samples: list[float] = []
    auc_samples: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yt = y_true_g[idx]
        yp = y_pred_g[idx]
        ypr = y_proba_g[idx]
        t, f, _, _, _, ppv_b = _confusion_rates(yt, yp)
        tpr_samples.append(t)
        fpr_samples.append(f)
        ppv_samples.append(ppv_b)
        if len(np.unique(yt)) >= 2:
            try:
                auc_samples.append(float(roc_auc_score(yt, ypr)))
            except Exception:
                pass

    return {
        "n": int(n),
        "tpr": _safe_float(tpr),
        "tpr_ci": list(_percentile_ci(tpr_samples)),
        "fpr": _safe_float(fpr),
        "fpr_ci": list(_percentile_ci(fpr_samples)),
        "fnr": _safe_float(fnr),
        "tnr": _safe_float(tnr),
        "ppv": _safe_float(ppv),
        "ppv_ci": list(_percentile_ci(ppv_samples)),
        "selection_rate": _safe_float(sr),
        "auc": _safe_float(auc),
        "auc_ci": list(_percentile_ci(auc_samples)) if auc_samples else [None, None],
        "sufficient_power": n >= N_THRESHOLD,
    }


def _compute_gaps(by_group: dict[str, dict]) -> dict:
    valid = [m for m in by_group.values() if m["n"] > 0]
    if len(valid) < 2:
        return {
            "tpr_gap": None, "fpr_gap": None,
            "eo_gap": None, "dp_gap": None, "di_ratio": None,
            "ppv_gap": None,
        }

    def _spread(field: str) -> float | None:
        xs = [m[field] for m in valid if m.get(field) is not None]
        return (max(xs) - min(xs)) if len(xs) >= 2 else None

    tpr_gap = _spread("tpr")
    fpr_gap = _spread("fpr")
    eo_components = [v for v in (tpr_gap, fpr_gap) if v is not None]
    eo_gap = max(eo_components) if eo_components else None
    dp_gap = _spread("selection_rate")
    ppv_gap = _spread("ppv")
    sr_values = [m["selection_rate"] for m in valid if m["selection_rate"] is not None]
    di_ratio = None
    if len(sr_values) >= 2 and max(sr_values) > 0:
        di_ratio = min(sr_values) / max(sr_values)

    return {
        "tpr_gap": _safe_float(tpr_gap),
        "fpr_gap": _safe_float(fpr_gap),
        "eo_gap": _safe_float(eo_gap),
        "dp_gap": _safe_float(dp_gap),
        "di_ratio": _safe_float(di_ratio),
        "ppv_gap": _safe_float(ppv_gap),
    }


# =========================================================
# Routes
# =========================================================

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/csv/inspect")
def inspect():
    if "file" not in request.files:
        return jsonify({"error": "missing 'file' upload"}), 400

    try:
        df = _read_csv_from_request()
    except Exception as e:
        return jsonify({"error": f"failed to parse CSV: {e}"}), 400

    columns = [_inspect_column(df[c]) for c in df.columns]
    return jsonify({"columns": columns, "row_count": int(len(df))})


@app.post("/api/audit/stage/1")
def stage_1():
    if "file" not in request.files:
        return jsonify({"error": "missing 'file' upload"}), 400

    target = (request.form.get("target") or "").strip()
    positive_class = request.form.get("positive_class")

    # Accept protected as repeated field, comma-separated, or single string.
    raw_protected = request.form.getlist("protected") or []
    if not raw_protected:
        single = request.form.get("protected")
        if single:
            raw_protected = [single]
    protected_list: list[str] = []
    for p in raw_protected:
        for piece in str(p).split(","):
            piece = piece.strip()
            if piece and piece not in protected_list:
                protected_list.append(piece)

    if not target or not protected_list or positive_class is None:
        return jsonify({"error": "missing required fields: target, protected, positive_class"}), 400

    try:
        df = _read_csv_from_request()
    except Exception as e:
        return jsonify({"error": f"failed to parse CSV: {e}"}), 400

    results = []
    for protected_col in protected_list:
        try:
            df_clean = preprocess(df, target, protected_col, positive_class)
            fingerprint = bias_fingerprint(df_clean, target, protected_col)
        except KeyError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"stage 1 failed for '{protected_col}': {e}"}), 500
        results.append({"protected": protected_col, "fingerprint": fingerprint})

    return jsonify({"results": results})


@app.post("/api/audit/stage/2")
def stage_2():
    """Stream Stage 2 results as ndjson. Trains all 5 models in parallel via
    a ThreadPoolExecutor; emits one event per model as it completes.

    Wire format (one JSON object per line):
        {"event": "init", "models": [...], "n_train": ..., ...meta}
        {"event": "model_done", "model": {...}}      (one per model)
        {"event": "done", "total_train_time_sec": ...}
    """
    if "file" not in request.files:
        return jsonify({"error": "missing 'file' upload"}), 400

    target = (request.form.get("target") or "").strip()
    positive_class = request.form.get("positive_class")

    raw_protected = request.form.getlist("protected") or []
    if not raw_protected:
        single = request.form.get("protected")
        if single:
            raw_protected = [single]
    protected_list: list[str] = []
    for p in raw_protected:
        for piece in str(p).split(","):
            piece = piece.strip()
            if piece and piece not in protected_list:
                protected_list.append(piece)

    if not target or positive_class is None:
        return jsonify({"error": "missing required fields: target, positive_class"}), 400

    try:
        df = _read_csv_from_request()
    except Exception as e:
        return jsonify({"error": f"failed to parse CSV: {e}"}), 400

    try:
        X, y, n_features, protected_values, feature_names = prepare_features(
            df, target, protected_list, positive_class
        )
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"feature prep failed: {e}"}), 500

    if len(np.unique(y)) < 2:
        return jsonify({
            "error": "target has fewer than 2 classes after encoding — check positive_class",
        }), 400

    n_train = int(len(y))
    n_features_out = int(n_features)
    session_id = uuid.uuid4().hex
    # Predictions accumulator — populated as each future resolves so we can
    # store them in SESSIONS once the stream finishes.
    session_predictions: dict[str, dict] = {}

    def stream():
        def line(obj: dict) -> bytes:
            return (json.dumps(obj) + "\n").encode("utf-8")

        init_models = [
            {
                "key": s["key"],
                "name": s["name"],
                "family": s["family"],
                "color": s["color"],
                "available": s["available"],
                "fallback_note": s["fallback_note"],
                "status": "running",
                "best_score": None,
                "best_params": {},
                "n_trials": 0,
                "cv_mean": None,
                "cv_std": None,
                "train_time_sec": None,
                "error": None,
            }
            for s in MODEL_REGISTRY
        ]
        yield line({
            "event": "init",
            "session_id": session_id,
            "models": init_models,
            "n_train": n_train,
            "n_features": n_features_out,
            "n_trials_per_model": N_TRIALS,
            "cv_folds": CV_FOLDS,
            "scoring": SCORING,
            "xgboost_available": HAS_XGB,
            "lightgbm_available": HAS_LGBM,
        })

        t0 = time.perf_counter()
        completed_models: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(MODEL_REGISTRY)) as ex:
            futures = {ex.submit(_train_one, spec, X, y): spec for spec in MODEL_REGISTRY}
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    result, y_pred, y_proba = fut.result()
                except Exception as e:
                    result = {
                        "key": spec["key"],
                        "name": spec["name"],
                        "family": spec["family"],
                        "color": spec["color"],
                        "available": spec["available"],
                        "fallback_note": spec["fallback_note"],
                        "n_trials": 0,
                        "best_score": None,
                        "best_params": {},
                        "cv_mean": None,
                        "cv_std": None,
                        "train_time_sec": None,
                        "status": "error",
                        "error": str(e),
                    }
                    y_pred, y_proba = None, None
                completed_models.append(result)
                if y_pred is not None and y_proba is not None:
                    session_predictions[spec["key"]] = {
                        "y_pred": y_pred,
                        "y_proba": y_proba,
                        "name": spec["name"],
                        "family": spec["family"],
                        "color": spec["color"],
                        "best_params": result.get("best_params", {}),
                        "best_score": result.get("best_score"),
                    }
                yield line({"event": "model_done", "model": result})

        # Cache the audit so Stages 3–5 can reuse it without re-training.
        _store_session(session_id, {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "protected_values": protected_values,
            "protected_names": list(protected_list),
            "target": target,
            "positive_class": str(positive_class),
            "predictions": session_predictions,
            "models": completed_models,
            "created_at": time.time(),
        })

        total = time.perf_counter() - t0
        yield line({"event": "done", "total_train_time_sec": _safe_float(total)})

    # Disable proxy buffering so events arrive promptly.
    headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    return Response(stream(), mimetype="application/x-ndjson", headers=headers)


def _stage3_compute_model_row(
    key: str,
    pred: dict,
    y: np.ndarray,
    protected_values: dict,
    model_overall: dict,
) -> list[dict]:
    """Compute the per-(attr, model) row used by Stage 3. Extracted so it can
    be parallelized across models when streaming."""
    rows = []
    y_pred = pred["y_pred"]
    y_proba = pred["y_proba"]
    for col, values in protected_values.items():
        groups_in_data = sorted(
            set(v for v in values if v and v.lower() not in {"nan", "none", "null"})
        )
        by_group: dict[str, dict] = {}
        for i, g in enumerate(groups_in_data):
            mask = values == g
            if mask.sum() == 0:
                continue
            by_group[g] = _group_metrics(y[mask], y_pred[mask], y_proba[mask], seed=42 + i)
        rows.append({
            "protected": col,
            "groups": groups_in_data,
            "model": {
                "key": key,
                "name": pred["name"],
                "family": pred["family"],
                "color": pred["color"],
                "overall_auc": model_overall[key]["overall_auc"],
                "overall_auc_ci": model_overall[key]["overall_auc_ci"],
                "ece": model_overall[key]["ece"],
                "by_group": by_group,
                "gaps": _compute_gaps(by_group),
            },
        })
    return rows


@app.post("/api/audit/stage/3")
def stage_3():
    """Per-model fairness audit. Reads cached predictions from the session
    written by Stage 2 — re-running Stage 2 issues a new session_id."""
    session_id = request.form.get("session_id") or request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id (run Stage 2 first)"}), 400
    sess = _get_session(session_id)
    if sess is None:
        return jsonify({"error": "session expired or not found — please re-run Stage 2"}), 404

    y: np.ndarray = sess["y"]
    protected_values: dict[str, np.ndarray] = sess["protected_values"]
    predictions: dict[str, dict] = sess["predictions"]

    if not predictions:
        return jsonify({"error": "no successful models in this session"}), 400

    # Per-model overall AUC + bootstrap CI + ECE — independent of protected attr.
    model_keys = list(predictions.keys())
    model_overall: dict[str, dict] = {}
    for key, pred in predictions.items():
        y_proba = pred["y_proba"]
        try:
            overall_auc = (
                float(roc_auc_score(y, y_proba))
                if len(np.unique(y)) >= 2 else None
            )
        except Exception:
            overall_auc = None
        ci_lo, ci_hi = _bootstrap_auc_ci(y, y_proba, BOOTSTRAP_N, seed=42)
        model_overall[key] = {
            "overall_auc": _safe_float(overall_auc),
            "overall_auc_ci": [ci_lo, ci_hi],
            "ece": _ece(y, y_proba),
        }

    # Pairwise statistical tests: McNemar (error disagreement) + DeLong (AUC diff).
    # BH FDR correction applied separately to each test family.
    pair_meta: list[tuple[str, str]] = []
    raw_mcnemar: list[float | None] = []
    raw_delong: list[float | None] = []
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            ka, kb = model_keys[i], model_keys[j]
            pair_meta.append((ka, kb))
            raw_mcnemar.append(
                _mcnemar_pvalue(y, predictions[ka]["y_pred"], predictions[kb]["y_pred"])
            )
            raw_delong.append(
                _delong_pvalue(y, predictions[ka]["y_proba"], predictions[kb]["y_proba"])
            )

    adj_mcnemar = _bh_fdr(raw_mcnemar)
    adj_delong = _bh_fdr(raw_delong)

    pairwise_tests = [
        {
            "model_a": ka,
            "model_b": kb,
            "mcnemar_p": mc,
            "mcnemar_p_adj": mc_adj,
            "delong_p": dl,
            "delong_p_adj": dl_adj,
            "significant_errors": mc_adj is not None and mc_adj < 0.05,
            "significant_auc": dl_adj is not None and dl_adj < 0.05,
        }
        for (ka, kb), mc, mc_adj, dl, dl_adj
        in zip(pair_meta, raw_mcnemar, adj_mcnemar, raw_delong, adj_delong)
    ]

    results = []
    for col, values in protected_values.items():
        groups_in_data = sorted(
            set(v for v in values if v and v.lower() not in {"nan", "none", "null"})
        )

        attr_models = []
        for key, pred in predictions.items():
            y_pred = pred["y_pred"]
            y_proba = pred["y_proba"]

            by_group: dict[str, dict] = {}
            for i, g in enumerate(groups_in_data):
                mask = values == g
                if mask.sum() == 0:
                    continue
                by_group[g] = _group_metrics(
                    y[mask], y_pred[mask], y_proba[mask],
                    seed=42 + i,
                )

            attr_models.append({
                "key": key,
                "name": pred["name"],
                "family": pred["family"],
                "color": pred["color"],
                "overall_auc": model_overall[key]["overall_auc"],
                "overall_auc_ci": model_overall[key]["overall_auc_ci"],
                "ece": model_overall[key]["ece"],
                "by_group": by_group,
                "gaps": _compute_gaps(by_group),
            })

        results.append({
            "protected": col,
            "groups": groups_in_data,
            "models": attr_models,
        })

    return jsonify({
        "session_id": session_id,
        "n_total": int(len(y)),
        "bootstrap_n": BOOTSTRAP_N,
        "pairwise_tests": pairwise_tests,
        "results": results,
    })


@app.post("/api/audit/stage/3/stream")
def stage_3_stream():
    """Streamed Stage 3 — emits one model_done event per model, then a
    pairwise_done event with all pairwise tests, then a done event.
    Mirrors Stage 2's streaming UX so models appear progressively."""
    session_id = request.form.get("session_id") or request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id (run Stage 2 first)"}), 400
    sess = _get_session(session_id)
    if sess is None:
        return jsonify({"error": "session expired or not found — please re-run Stage 2"}), 404

    y: np.ndarray = sess["y"]
    protected_values: dict[str, np.ndarray] = sess["protected_values"]
    predictions: dict[str, dict] = sess["predictions"]

    if not predictions:
        return jsonify({"error": "no successful models in this session"}), 400

    # Pre-compute overall AUC + bootstrap CI + ECE upfront so model_done events
    # can carry them when each model's per-group fairness audit lands.
    model_keys = list(predictions.keys())
    model_overall: dict[str, dict] = {}
    for key, pred in predictions.items():
        y_proba = pred["y_proba"]
        try:
            overall_auc = (
                float(roc_auc_score(y, y_proba))
                if len(np.unique(y)) >= 2 else None
            )
        except Exception:
            overall_auc = None
        ci_lo, ci_hi = _bootstrap_auc_ci(y, y_proba, BOOTSTRAP_N, seed=42)
        model_overall[key] = {
            "overall_auc": _safe_float(overall_auc),
            "overall_auc_ci": [ci_lo, ci_hi],
            "ece": _ece(y, y_proba),
        }

    # Skeleton of "running" placeholders, one per (attr, model).
    init_results = []
    for col, values in protected_values.items():
        groups_in_data = sorted(
            set(v for v in values if v and v.lower() not in {"nan", "none", "null"})
        )
        init_results.append({
            "protected": col,
            "groups": groups_in_data,
            "models": [
                {
                    "key": k,
                    "name": predictions[k]["name"],
                    "family": predictions[k]["family"],
                    "color": predictions[k]["color"],
                    "status": "running",
                }
                for k in model_keys
            ],
        })

    def stream():
        def line(obj: dict) -> bytes:
            return (json.dumps(obj) + "\n").encode("utf-8")

        yield line({
            "event": "init",
            "session_id": session_id,
            "n_total": int(len(y)),
            "bootstrap_n": BOOTSTRAP_N,
            "results": init_results,
        })

        # Run per-model audits sequentially so each model_done event lands as a
        # separate flush and the UI can render models progressively. Bootstrap
        # CIs are GIL-bound dict-heavy work — sequential is comparable in total
        # time to the parallel version on typical hackathon-scale datasets and
        # gives the user a visible "one at a time" streaming UX that matches Stage 2.
        for k in model_keys:
            try:
                rows = _stage3_compute_model_row(
                    k, predictions[k], y, protected_values, model_overall
                )
            except Exception as e:
                yield line({
                    "event": "model_done",
                    "model_key": k,
                    "error": str(e),
                })
                continue
            for row in rows:
                yield line({
                    "event": "model_done",
                    "protected": row["protected"],
                    "model_key": k,
                    "model": row["model"],
                })

        # Pairwise tests — cheap, depends on raw predictions which are already in memory.
        pair_meta: list[tuple[str, str]] = []
        raw_mcnemar: list[float | None] = []
        raw_delong: list[float | None] = []
        for i in range(len(model_keys)):
            for j in range(i + 1, len(model_keys)):
                ka, kb = model_keys[i], model_keys[j]
                pair_meta.append((ka, kb))
                raw_mcnemar.append(_mcnemar_pvalue(y, predictions[ka]["y_pred"], predictions[kb]["y_pred"]))
                raw_delong.append(_delong_pvalue(y, predictions[ka]["y_proba"], predictions[kb]["y_proba"]))
        adj_mcnemar = _bh_fdr(raw_mcnemar)
        adj_delong = _bh_fdr(raw_delong)
        pairwise_tests = [
            {
                "model_a": ka,
                "model_b": kb,
                "mcnemar_p": mc,
                "mcnemar_p_adj": mc_adj,
                "delong_p": dl,
                "delong_p_adj": dl_adj,
                "significant_errors": mc_adj is not None and mc_adj < 0.05,
                "significant_auc": dl_adj is not None and dl_adj < 0.05,
            }
            for (ka, kb), mc, mc_adj, dl, dl_adj
            in zip(pair_meta, raw_mcnemar, adj_mcnemar, raw_delong, adj_delong)
        ]

        yield line({"event": "pairwise_done", "tests": pairwise_tests})
        yield line({"event": "done"})

    headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    return Response(stream(), mimetype="application/x-ndjson", headers=headers)


@app.post("/api/audit/stage/4")
def stage_4():
    """Pareto frontier per protected attribute. AUC vs equalized-odds gap.
    A model is dominated when another is at least as good on both axes and
    strictly better on one."""
    session_id = request.form.get("session_id") or request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id (run Stage 2 first)"}), 400
    sess = _get_session(session_id)
    if sess is None:
        return jsonify({"error": "session expired or not found — please re-run Stage 2"}), 404

    y: np.ndarray = sess["y"]
    protected_values: dict[str, np.ndarray] = sess["protected_values"]
    predictions: dict[str, dict] = sess["predictions"]

    if not predictions:
        return jsonify({"error": "no successful models in this session"}), 400

    # Reuse Stage 3 maths to keep accuracy/fairness numbers consistent
    # between the leaderboard and the Pareto chart.
    overall_auc: dict[str, float | None] = {}
    for key, pred in predictions.items():
        try:
            overall_auc[key] = (
                float(roc_auc_score(y, pred["y_proba"]))
                if len(np.unique(y)) >= 2 else None
            )
        except Exception:
            overall_auc[key] = None

    # Allow callers to override the threshold; default 0.15.
    try:
        eo_threshold = float(request.form.get("eo_gap_threshold") or EO_GAP_THRESHOLD)
    except (TypeError, ValueError):
        eo_threshold = EO_GAP_THRESHOLD
    try:
        lambda_param = float(request.form.get("lambda_param") or COMPOSITE_LAMBDA)
    except (TypeError, ValueError):
        lambda_param = COMPOSITE_LAMBDA

    results = []
    for col, values in protected_values.items():
        groups_in_data = sorted(
            set(v for v in values if v and v.lower() not in {"nan", "none", "null"})
        )

        rows = []
        for key, pred in predictions.items():
            y_pred = pred["y_pred"]
            y_proba = pred["y_proba"]
            by_group: dict[str, dict] = {}
            for i, g in enumerate(groups_in_data):
                mask = values == g
                if mask.sum() == 0:
                    continue
                # Skip bootstrap here — Stage 4 only needs point estimates.
                tpr, fpr, fnr, tnr, sr, ppv = _confusion_rates(y[mask], y_pred[mask])
                by_group[g] = {
                    "n": int(mask.sum()),
                    "tpr": _safe_float(tpr),
                    "fpr": _safe_float(fpr),
                    "fnr": _safe_float(fnr),
                    "tnr": _safe_float(tnr),
                    "selection_rate": _safe_float(sr),
                    "ppv": _safe_float(ppv),
                }
            gaps = _compute_gaps(by_group)
            auc_val = _safe_float(overall_auc[key])
            eo_val = gaps["eo_gap"]
            # Composite score: Score = AUC − λ·EO_gap. Higher = better tradeoff.
            composite = (
                auc_val - lambda_param * eo_val
                if auc_val is not None and eo_val is not None else None
            )
            rows.append({
                "key": key,
                "name": pred["name"],
                "family": pred["family"],
                "color": pred["color"],
                "auc": auc_val,
                "fairness_gap": eo_val,
                "tpr_gap": gaps["tpr_gap"],
                "fpr_gap": gaps["fpr_gap"],
                "dp_gap": gaps["dp_gap"],
                "di_ratio": gaps["di_ratio"],
                "ppv_gap": gaps["ppv_gap"],
                "composite_score": _safe_float(composite),
            })

        # Step 1 — Pareto dominance: r is dominated if some other row has
        # auc >= r.auc AND fairness_gap <= r.fairness_gap, with at least one strict.
        for r in rows:
            r["pareto_optimal"] = (
                r["auc"] is not None
                and r["fairness_gap"] is not None
                and not any(
                    o is not r
                    and o["auc"] is not None
                    and o["fairness_gap"] is not None
                    and o["auc"] >= r["auc"]
                    and o["fairness_gap"] <= r["fairness_gap"]
                    and (o["auc"] > r["auc"] or o["fairness_gap"] < r["fairness_gap"])
                    for o in rows
                )
            )

        # Step 2 — Fairness guardrail: a model satisfies the fairness constraint
        # iff its EO gap is at or below the threshold. This is independent of
        # Pareto-optimality and is computed for *every* model so the UI can
        # explain why a high-AUC model wasn't picked.
        for r in rows:
            r["fairness_qualified"] = (
                r["fairness_gap"] is not None
                and r["fairness_gap"] <= eo_threshold
            )

        # Step 3 — Selection rule: highest AUC among models that are Pareto-optimal
        # AND satisfy the fairness threshold. If no model qualifies, no recommendation
        # is issued and a warning surfaces so the user knows why.
        eligible = [r for r in rows if r["pareto_optimal"] and r["fairness_qualified"] and r["auc"] is not None]
        recommendation_warning = None
        if eligible:
            best = max(eligible, key=lambda r: r["auc"])
            for r in rows:
                r["recommended"] = r is best
            r["recommended_reason"] = (
                f"Highest AUC ({best['auc']:.3f}) among Pareto-optimal models with "
                f"EO gap ≤ {eo_threshold:.2f}"
            ) if False else None  # reason carried at attr-level below
            attr_reason = (
                f"Highest AUC ({best['auc']:.3f}) on the Pareto frontier with "
                f"equalized-odds gap {best['fairness_gap'] * 100:.1f}pp "
                f"(within {eo_threshold * 100:.0f}pp threshold)"
            )
        else:
            for r in rows:
                r["recommended"] = False
            attr_reason = None
            pareto_rows = [r for r in rows if r["pareto_optimal"]]
            if pareto_rows:
                # There are Pareto-optimal models, but none satisfy the fairness threshold.
                lowest_eo = min(
                    pareto_rows,
                    key=lambda r: r["fairness_gap"] if r["fairness_gap"] is not None else 1.0,
                )
                recommendation_warning = (
                    f"No model satisfies the fairness threshold "
                    f"(EO gap ≤ {eo_threshold * 100:.0f}pp). "
                    f"The fairest Pareto-optimal candidate is {lowest_eo['name']} "
                    f"(EO gap {lowest_eo['fairness_gap'] * 100:.1f}pp) — manual review required."
                )
            else:
                recommendation_warning = (
                    "No Pareto-optimal model could be identified. Inspect the trained "
                    "models individually before deploying."
                )

        results.append({
            "protected": col,
            "models": rows,
            "recommended_reason": attr_reason,
            "recommendation_warning": recommendation_warning,
        })

    return jsonify({
        "session_id": session_id,
        "eo_gap_threshold": eo_threshold,
        "lambda_param": lambda_param,
        "results": results,
    })


# =========================================================
# STAGE 5 — Root cause diagnosis
# =========================================================

def _shap_normalize(sv) -> np.ndarray | None:
    """Normalize SHAP output across versions to a (n_samples, n_features) 2D array.

    - Older shap (<0.46) returned a list [neg_class, pos_class] for binary classifiers.
    - Newer shap (>=0.46) returns a 3D array (n_samples, n_features, n_classes).
    Both cases are reduced to the positive-class slice.
    """
    if sv is None:
        return None
    if isinstance(sv, list):
        # Pick the positive class (index 1 for binary; last entry as a safe default).
        sv = sv[1] if len(sv) >= 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) — slice positive class for binary.
        sv = sv[..., 1] if sv.shape[-1] == 2 else sv[..., -1]
    if sv.ndim != 2:
        return None
    return sv


def _shap_values(key: str, model, X_train: np.ndarray, X_explain: np.ndarray) -> np.ndarray | None:
    """Compute SHAP values for X_explain. Returns (n_explain, n_features) 2D array or None."""
    if not HAS_SHAP:
        return None
    try:
        # Unwrap Pipeline: apply transforms, isolate final estimator.
        if hasattr(model, "steps"):
            X_exp_t, X_tr_t = X_explain.copy(), X_train.copy()
            estimator = None
            for _, step in model.steps:
                if hasattr(step, "transform"):
                    X_exp_t = step.transform(X_exp_t)
                    X_tr_t = step.transform(X_tr_t)
                else:
                    estimator = step
        else:
            X_exp_t, X_tr_t, estimator = X_explain, X_train, model

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_tr_t), size=min(SHAP_BG_N, len(X_tr_t)), replace=False)
        X_bg = X_tr_t[bg_idx]

        if isinstance(estimator, (RandomForestClassifier, HistGradientBoostingClassifier)):
            exp = shap_lib.TreeExplainer(estimator)
            return _shap_normalize(exp.shap_values(X_exp_t))

        if HAS_XGB and isinstance(estimator, XGBClassifier):
            exp = shap_lib.TreeExplainer(estimator)
            return _shap_normalize(exp.shap_values(X_exp_t))

        if HAS_LGBM and isinstance(estimator, LGBMClassifier):
            exp = shap_lib.TreeExplainer(estimator)
            return _shap_normalize(exp.shap_values(X_exp_t))

        if isinstance(estimator, LogisticRegression):
            exp = shap_lib.LinearExplainer(estimator, X_bg)
            return _shap_normalize(exp.shap_values(X_exp_t))

        if isinstance(estimator, LinearSVC):
            # Linear attribution: coef × standardized feature value (valid for linear models)
            return _shap_normalize(X_exp_t * estimator.coef_[0])

    except Exception:
        pass
    return None


def _group_shap_importance(
    sv: np.ndarray, mask: np.ndarray, feature_names: list[str]
) -> dict:
    """Mean |SHAP| importance and ordinal rank for top S5_TOP_N features in a group."""
    sv_g = np.abs(sv[mask])
    if len(sv_g) == 0:
        return {"importance": {}, "ranks": {}}
    mean_imp = np.asarray(sv_g.mean(axis=0)).ravel()  # guarantee 1D
    order = np.argsort(mean_imp)[::-1]
    top = [int(i) for i in order[:S5_TOP_N]]
    return {
        "importance": {feature_names[i]: _safe_float(float(mean_imp[i])) for i in top},
        "ranks": {feature_names[i]: r + 1 for r, i in enumerate(top)},
    }


def _proxy_features(
    group_shap: dict[str, dict], groups: list[str], ratio_threshold: float = 2.0
) -> list[dict]:
    """Features where SHAP importance differs ≥ratio_threshold× across groups."""
    if len(groups) < 2:
        return []
    all_feats = set()
    for g in groups:
        all_feats.update(group_shap.get(g, {}).get("importance", {}).keys())

    results = []
    for feat in all_feats:
        imp = {g: group_shap.get(g, {}).get("importance", {}).get(feat) for g in groups}
        valid = {g: v for g, v in imp.items() if v is not None and v > 0}
        if len(valid) < 2:
            continue
        max_g = max(valid, key=valid.get)
        min_g = min(valid, key=valid.get)
        ratio = valid[max_g] / max(valid[min_g], 1e-9)
        if ratio < ratio_threshold:
            continue
        rk_max = group_shap.get(max_g, {}).get("ranks", {}).get(feat)
        rk_min = group_shap.get(min_g, {}).get("ranks", {}).get(feat)
        results.append({
            "feature": feat,
            "ratio": _safe_float(ratio),
            "max_group": max_g,
            "min_group": min_g,
            "rank_delta": int(rk_min - rk_max) if rk_max and rk_min else None,
            "importances": {g: _safe_float(v) for g, v in imp.items() if v is not None},
        })
    results.sort(key=lambda x: -(x["ratio"] or 0))
    return results[:S5_TOP_N]


def _feature_group_correlation(
    X: np.ndarray, values: np.ndarray, groups: list[str], feature_names: list[str]
) -> list[dict]:
    """Standardized mean difference per feature between the two largest groups.
    Always computable (no SHAP required) — measures structural data bias."""
    if len(groups) < 2:
        return []
    ga, gb = groups[0], groups[1]
    mask_a = values == ga
    mask_b = values == gb
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return []
    mean_a = X[mask_a].mean(axis=0)
    mean_b = X[mask_b].mean(axis=0)
    std = X.std(axis=0) + 1e-9
    smd = np.abs(mean_a - mean_b) / std
    order = np.argsort(smd)[::-1][:S5_TOP_N]
    return [
        {
            "feature": feature_names[i],
            "smd": _safe_float(float(smd[i])),
            "mean_a": _safe_float(float(mean_a[i])),
            "mean_b": _safe_float(float(mean_b[i])),
            "group_a": ga,
            "group_b": gb,
        }
        for i in order
    ]


def _permutation_pvalue(
    sv: np.ndarray, group_values_sub: np.ndarray, groups: list[str], n_perm: int = PERM_N
) -> float | None:
    """Permutation test: H0 = group label has no effect on SHAP rank structure.
    Observed stat = max rank delta between the two largest groups."""
    if sv is None or len(groups) < 2:
        return None
    ga, gb = groups[0], groups[1]

    def _stat(labels: np.ndarray) -> float:
        ma, mb = labels == ga, labels == gb
        if ma.sum() == 0 or mb.sum() == 0:
            return 0.0
        ia = np.abs(sv[ma]).mean(axis=0)
        ib = np.abs(sv[mb]).mean(axis=0)
        ra = np.argsort(np.argsort(ia)[::-1])
        rb = np.argsort(np.argsort(ib)[::-1])
        return float(np.max(np.abs(ra - rb)))

    observed = _stat(group_values_sub)
    rng = np.random.default_rng(42)
    count = sum(
        1 for _ in range(n_perm) if _stat(rng.permutation(group_values_sub)) >= observed
    )
    return _safe_float(count / n_perm)


def _flip_rate(
    model, X: np.ndarray, values: np.ndarray, groups: list[str],
    y_pred_orig: np.ndarray, n_top: int = 3,
) -> float | None:
    """Counterfactual flip rate: replace top group-discriminating features with
    the other group's mean, repredict, count the fraction of predictions that change."""
    if len(groups) < 2:
        return None
    ga, gb = groups[0], groups[1]
    mask_a, mask_b = values == ga, values == gb
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None

    mean_a, mean_b = X[mask_a].mean(axis=0), X[mask_b].mean(axis=0)
    smd = np.abs(mean_a - mean_b) / (X.std(axis=0) + 1e-9)
    top_idx = np.argsort(smd)[::-1][:n_top]

    total_flips = total_n = 0
    for src_mask, tgt_mean in [(mask_a, mean_b), (mask_b, mean_a)]:
        if src_mask.sum() == 0:
            continue
        X_mod = X[src_mask].copy()
        for i in top_idx:
            X_mod[:, i] = tgt_mean[i]
        try:
            new_pred = model.predict(X_mod)
        except Exception:
            continue
        total_flips += int((new_pred != y_pred_orig[src_mask]).sum())
        total_n += int(src_mask.sum())

    return _safe_float(total_flips / total_n) if total_n > 0 else None


def _bayesian_posterior(
    proxy_score: float, flip_rate: float, group_imbalance: float,
    base_rate_gap: float, eo_gap: float, dp_gap: float,
) -> dict:
    """Heuristic Bayesian posterior over 5 bias classes. Scores are unnormalized then normalized."""
    proxy       = max(proxy_score - 1.0, 0.0) ** 1.5 * (1.0 + flip_rate * 2.0)
    represent   = max(group_imbalance - 1.0, 0.0) * base_rate_gap * 2.0
    label       = base_rate_gap * max(1.0 - flip_rate, 0.0)
    threshold   = max(dp_gap - eo_gap, 0.0) * 3.0
    complexity  = eo_gap * max(1.0 - flip_rate, 0.0) * 0.5
    scores = {
        "proxy_discrimination": proxy,
        "representation_bias": represent,
        "label_bias": label,
        "threshold_effect": threshold,
        "model_complexity_bias": complexity,
    }
    total = sum(scores.values()) + 1e-9
    return {k: _safe_float(v / total) for k, v in scores.items()}


@app.post("/api/audit/stage/5")
def stage_5():
    """Root cause diagnosis — SHAP per group, proxy detection, permutation test,
    counterfactual flip rate, Bayesian 5-class posterior."""
    session_id = request.form.get("session_id") or request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id (run Stage 2 first)"}), 400
    sess = _get_session(session_id)
    if sess is None:
        return jsonify({"error": "session expired — please re-run Stage 2"}), 404

    model_key = (request.form.get("model_key") or "").strip() or None
    X: np.ndarray = sess["X"]
    y: np.ndarray = sess["y"]
    protected_values: dict[str, np.ndarray] = sess["protected_values"]
    predictions: dict[str, dict] = sess["predictions"]
    feature_names: list[str] = sess.get("feature_names") or [f"f{i}" for i in range(X.shape[1])]

    if not predictions:
        return jsonify({"error": "no successful models in session"}), 400

    # Select model — default to highest Optuna AUC (closest to Pareto recommendation)
    if model_key is None or model_key not in predictions:
        model_key = max(predictions, key=lambda k: predictions[k].get("best_score") or 0.0)

    pred_meta = predictions[model_key]

    # Refit on full data so SHAP and predict() work outside CV folds.
    try:
        fitted = _build_model_from_params(model_key, pred_meta.get("best_params", {}))
        fitted.fit(X, y)
    except Exception as e:
        return jsonify({"error": f"model refit failed: {e}"}), 500

    # Subsample X for SHAP (keeps latency acceptable on large datasets).
    rng = np.random.default_rng(42)
    explain_idx = rng.choice(len(X), size=min(SHAP_EXPLAIN_N, len(X)), replace=False)
    X_sub = X[explain_idx]

    sv = _shap_values(model_key, fitted, X, X_sub)  # (n_explain, n_features) or None

    results: dict = {}
    for col, values in protected_values.items():
        groups = sorted(
            set(v for v in values if v and v.lower() not in {"nan", "none", "null"})
        )
        if len(groups) < 2:
            continue

        values_sub = values[explain_idx]  # group labels aligned with X_sub / sv
        y_pred_orig = pred_meta["y_pred"]  # OOF predictions from Stage 2

        # ── SHAP per-group analysis ──────────────────────────────────────
        group_shap: dict[str, dict] = {}
        if sv is not None:
            for g in groups:
                mask_sub = values_sub == g
                if mask_sub.sum() > 0:
                    group_shap[g] = _group_shap_importance(sv, mask_sub, feature_names)

        proxy_feats = _proxy_features(group_shap, groups) if group_shap else []
        proxy_score = proxy_feats[0]["ratio"] if proxy_feats else 1.0

        # ── Feature–group correlation (always computed) ──────────────────
        corr_feats = _feature_group_correlation(X, values, groups, feature_names)

        # ── Permutation test ─────────────────────────────────────────────
        perm_p = _permutation_pvalue(sv, values_sub, groups) if sv is not None else None

        # ── Counterfactual flip rate ─────────────────────────────────────
        fr = _flip_rate(fitted, X, values, groups, y_pred_orig)
        fr_val = fr if fr is not None else 0.0

        # ── Per-group stats from OOF predictions ─────────────────────────
        tpr_list, sr_list, pos_list, ns = [], [], [], []
        for g in groups:
            mask_g = values == g
            if mask_g.sum() == 0:
                continue
            tpr_g, _, _, _, sr_g, _ = _confusion_rates(y[mask_g], y_pred_orig[mask_g])
            tpr_list.append(tpr_g)
            sr_list.append(sr_g)
            pos_list.append(float(y[mask_g].mean()))
            ns.append(int(mask_g.sum()))

        eo_gap  = (max(tpr_list) - min(tpr_list))  if len(tpr_list) >= 2 else 0.0
        dp_gap  = (max(sr_list)  - min(sr_list))   if len(sr_list)  >= 2 else 0.0
        br_gap  = (max(pos_list) - min(pos_list))  if len(pos_list) >= 2 else 0.0
        imbal   = (max(ns) / max(min(ns), 1))      if ns else 1.0

        posterior = _bayesian_posterior(
            float(proxy_score), float(fr_val), float(imbal),
            float(br_gap), float(eo_gap), float(dp_gap),
        )

        # Per-group sample counts and positive rates — used by the frontend to
        # render forensic-style explanations ("Group X has only N samples vs M").
        group_sizes = {g: int((values == g).sum()) for g in groups}
        group_pos_rates = {
            g: _safe_float(float(y[values == g].mean()))
            if (values == g).sum() > 0 else None
            for g in groups
        }

        results[col] = {
            "groups": groups,
            "group_sizes": group_sizes,
            "group_positive_rates": group_pos_rates,
            "group_shap": group_shap,
            "proxy_features": proxy_feats,
            "correlated_features": corr_feats,
            "permutation_test": {
                "p_value": perm_p,
                "n_permutations": PERM_N,
                "significant": perm_p is not None and perm_p < 0.05,
            },
            "counterfactual_flip_rate": _safe_float(fr_val),
            "eo_gap": _safe_float(eo_gap),
            "dp_gap": _safe_float(dp_gap),
            "base_rate_gap": _safe_float(br_gap),
            "bayesian_root_cause": posterior,
        }

    primary_posterior = next(iter(results.values()), {}).get("bayesian_root_cause") if results else None
    primary_cause = (
        max(primary_posterior, key=lambda k: primary_posterior[k] or 0)
        if primary_posterior else None
    )

    return jsonify({
        "session_id": session_id,
        "model_key": model_key,
        "model_name": pred_meta.get("name", model_key),
        "shap_available": HAS_SHAP and sv is not None,
        "primary_root_cause": primary_cause,
        "results": results,
    })


# =========================================================
# STAGE 6 — Guided remediation
# =========================================================

def _remediation_plan(
    primary_root_cause: str,
    confidence: float,
    eo_gap: float,
    dp_gap: float,
    flip_rate: float,
    proxy_count: int,
    top_proxy_feature: str | None,
    model_name: str,
    protected_attr: str | None,
    *,
    minority_label: str | None = None,
    majority_label: str | None = None,
    minority_n: int | None = None,
    majority_n: int | None = None,
    attr_human_label: str | None = None,
) -> dict:
    """Deterministic conditional remediation engine.

    Safety principle: prefer no action over harmful action.
    Never block a legitimate fix; never approve an unsafe one.

    Specificity inputs (minority_label, majority_label, etc.) come from the
    frontend's encoding-aware label decoder so recommendations name the actual
    group rather than abstract "the minority group".
    """
    attr_label = attr_human_label or protected_attr or "the protected attribute"
    feat_label = top_proxy_feature or "the correlated feature"

    # Build a specific group reference if we have it; otherwise generic.
    if minority_label and majority_label and minority_n is not None and majority_n is not None:
        ratio = majority_n / max(minority_n, 1)
        group_specific = (
            f"{minority_label} (n = {minority_n}) vs. {majority_label} (n = {majority_n}) — "
            f"{ratio:.1f}× more samples in the majority group"
        )
        group_short = f"{minority_label} (n = {minority_n})"
    else:
        group_specific = "the minority group"
        group_short = "the minority group"

    if primary_root_cause == "proxy_discrimination":
        actions = [
            {
                "id": "feature_decorrelation",
                "title": f"Decorrelate {feat_label} from {attr_label}",
                "status": "recommended",
                "body": (
                    f"SHAP analysis identifies {feat_label} as the primary proxy for {attr_label}. "
                    f"Counterfactual flip rate {flip_rate * 100:.1f}% confirms a causal pathway "
                    f"through this feature in {model_name}. Remove or orthogonalize "
                    f"{feat_label} before retraining."
                ),
                "success_criteria": [
                    "EO gap decreases by ≥ 30%",
                    "AUC drops by no more than 1.0pp",
                    "Counterfactual flip rate falls below 10%",
                ],
            },
            {
                "id": "group_calibration",
                "title": "Group-calibrated probability scaling",
                "status": "optional",
                "body": (
                    "Platt scaling applied per group can reduce ECE without altering the feature set. "
                    "Bootstrap-validate that the ECE improvement is a genuine Pareto gain "
                    "and not trading group accuracy for apparent fairness."
                ),
                "success_criteria": [
                    "Per-group ECE drops by ≥ 25%",
                    "AUC unchanged (within ±0.5pp)",
                ],
            },
            {
                "id": "threshold_adjustment",
                "title": "Group-specific threshold adjustment",
                "status": "blocked",
                "body": (
                    f"Root cause is proxy discrimination, not score miscalibration. "
                    f"Adjusting decision thresholds per group on {model_name} would mask the "
                    "upstream mechanism, not fix it, and could introduce disparate-treatment "
                    "legal liability. Blocked by the remediation engine."
                ),
                "success_criteria": [],
            },
        ]
        return {
            "diagnosis": "Proxy discrimination via feature correlation",
            "summary": (
                f"{model_name} exhibits an equalized-odds gap of {eo_gap * 100:.1f}pp driven by "
                f"{feat_label}, which is statistically correlated with {attr_label}. "
                f"Bayesian posterior assigns {confidence * 100:.0f}% probability to this cause class."
            ),
            "actions": actions,
            "safe_to_auto_fix": False,
            "warning": (
                f"Domain expertise required to verify that removing or orthogonalizing "
                f"{feat_label} does not eliminate legitimate predictive signal for {model_name}."
            ),
        }

    elif primary_root_cause == "threshold_effect":
        safe = confidence >= 0.70
        actions: list[dict] = [
            {
                "id": "threshold_optimization",
                "title": f"Group-aware threshold optimization for {model_name}",
                "status": "recommended",
                "body": (
                    f"DP gap ({dp_gap * 100:.1f}pp) substantially exceeds EO gap ({eo_gap * 100:.1f}pp) "
                    f"on {model_name}, indicating the 0.5 decision threshold produces unequal "
                    f"selection rates between {group_short if minority_label else 'the two groups'} "
                    "from otherwise comparable score distributions. Optimize per-group thresholds "
                    "via ROC analysis and validate as a Pareto improvement."
                ),
                "success_criteria": [
                    "DP gap drops to within 5pp of EO gap",
                    "AUC drops by no more than 0.5pp",
                    "Both groups maintain ≥ 90% of their original recall",
                ],
            },
            {
                "id": "isotonic_calibration",
                "title": "Isotonic calibration",
                "status": "optional",
                "body": (
                    "Isotonic regression calibration aligns probability scores with observed "
                    "frequencies per group, reducing the calibration error that drives the "
                    f"threshold disparity in {model_name}."
                ),
                "success_criteria": [
                    "Per-group ECE drops by ≥ 25%",
                    "AUC unchanged (within ±0.5pp)",
                ],
            },
        ]
        if proxy_count > 0:
            actions.append({
                "id": "proxy_check",
                "title": "Verify proxy features are secondary",
                "status": "optional",
                "body": (
                    f"{proxy_count} potential proxy feature{'s' if proxy_count > 1 else ''} detected. "
                    "Confirm these are not a primary contributor before applying threshold fixes alone; "
                    "if proxy score is high, address feature decorrelation first."
                ),
                "success_criteria": [
                    "Proxy SHAP-importance ratio falls below 2.0×",
                ],
            })
        return {
            "diagnosis": "Threshold effect — disparate outcomes from score miscalibration",
            "summary": (
                f"On {model_name}, DP gap ({dp_gap * 100:.1f}pp) exceeds EO gap ({eo_gap * 100:.1f}pp) "
                f"with {confidence * 100:.0f}% confidence. The 0.5 decision threshold is not "
                f"group-neutral across {attr_label}; adjusting it is the primary lever."
            ),
            "actions": actions,
            "safe_to_auto_fix": safe,
            "warning": None if safe else (
                f"Confidence {confidence * 100:.0f}% is below the 70% auto-apply threshold. "
                "Manual review recommended before deploying threshold changes."
            ),
        }

    elif primary_root_cause == "label_bias":
        actions = [
            {
                "id": "human_review",
                "title": f"Human review of labeling process for {attr_label}",
                "status": "recommended",
                "body": (
                    f"The labels in this dataset carry embedded bias that {model_name} learned directly. "
                    f"With a low counterfactual flip rate ({flip_rate * 100:.1f}%), group differences "
                    "are encoded in the ground truth itself rather than the feature space. "
                    "Algorithmic interventions cannot safely fix label bias without first "
                    "understanding the original decision-making process."
                ),
                "success_criteria": [
                    "Documented review of label generation procedure",
                    "Sign-off from domain expert on labeling consistency",
                ],
            },
            {
                "id": "label_audit",
                "title": "Stratified label audit with domain experts",
                "status": "recommended",
                "body": (
                    f"Review a random stratified sample of labels for {group_specific} with domain experts. "
                    "Estimate the counterfactual positive rate if historical decisions had been "
                    "made without access to group membership. Document findings before any retraining."
                ),
                "success_criteria": [
                    "≥ 50 labels per group reviewed",
                    "Inter-rater agreement κ ≥ 0.7",
                ],
            },
            {
                "id": "algorithmic_fix",
                "title": "Algorithmic fairness intervention",
                "status": "blocked",
                "body": (
                    "No in-processing or post-processing algorithmic fix is appropriate when the "
                    f"primary cause is label bias. Applying SMOTE, threshold tuning, or reweighting on "
                    f"{model_name} would optimize for a biased target and produce misleading fairness "
                    "metrics without addressing the root problem. Blocked by the remediation engine."
                ),
                "success_criteria": [],
            },
        ]
        return {
            "diagnosis": "Label bias — historical decisions embedded in training labels",
            "summary": (
                f"Primary root cause is label bias ({confidence * 100:.0f}% confidence). "
                f"The bias originates upstream in the data generation process for {attr_label}. "
                f"No automated algorithmic fix is safe for {model_name} without human review "
                "of the labeling criteria."
            ),
            "actions": actions,
            "safe_to_auto_fix": False,
            "warning": (
                f"All algorithmic interventions are blocked. Deploying {model_name} in its current "
                "form risks perpetuating historical discrimination. Human intervention is required "
                "before any deployment decision."
            ),
        }

    elif primary_root_cause == "representation_bias":
        safe = confidence >= 0.70
        actions = [
            {
                "id": "resampling",
                "title": (
                    f"Resample {minority_label or 'minority group'} with SMOTE / class weighting"
                    if minority_label else "Stratified resampling or class weighting"
                ),
                "status": "recommended" if safe else "optional",
                "body": (
                    f"{group_specific.capitalize()} is causing {model_name} to learn decision "
                    f"boundaries primarily from the majority group. "
                    "Apply stratified oversampling (SMOTE) or inverse-frequency class weights "
                    "and retrain."
                ),
                "success_criteria": [
                    "EO gap decreases by ≥ 30%",
                    "AUC drops by no more than 1.0pp",
                    f"Recall on {minority_label or 'the minority group'} improves by ≥ 5pp",
                ],
            },
            {
                "id": "data_collection",
                "title": (
                    f"Collect additional data for {minority_label or 'the underrepresented group'}"
                    if minority_label else "Collect additional data for underrepresented groups"
                ),
                "status": "optional",
                "body": (
                    f"If resampling alone is insufficient, real data collection for "
                    f"{minority_label or 'underrepresented groups'} is the most robust long-term fix. "
                    "Synthetic augmentation (SMOTE, GAN) is a short-term alternative with "
                    "known risks of distribution shift."
                ),
                "success_criteria": [
                    f"{minority_label or 'Minority group'} sample count reaches at least "
                    f"{int((majority_n or 100) * 0.3)} (≥ 30% of majority)"
                    if majority_n else "Minority group reaches ≥ 30% of majority size",
                ],
            },
        ]
        return {
            "diagnosis": "Representation bias — training data group imbalance",
            "summary": (
                f"Group imbalance is the primary driver ({confidence * 100:.0f}% confidence). "
                f"{model_name} has learned skewed decision boundaries due to insufficient samples "
                f"in {attr_label}. {group_specific.capitalize()}."
            ),
            "actions": actions,
            "safe_to_auto_fix": safe,
            "warning": None if safe else (
                f"Confidence {confidence * 100:.0f}% is below the 70% auto-apply threshold. "
                "Manual review recommended before applying resampling interventions."
            ),
        }

    else:  # model_complexity_bias or unknown
        actions = [
            {
                "id": "regularization",
                "title": f"Increase regularization or simplify {model_name}",
                "status": "recommended",
                "body": (
                    f"{model_name} may be overfitting group-specific noise. "
                    "Increase L2 regularization strength or switch to a simpler model family "
                    "(e.g., Logistic Regression) and compare fairness metrics under equivalent "
                    "Optuna hyperparameter budget."
                ),
                "success_criteria": [
                    "EO gap decreases by ≥ 20%",
                    "Out-of-fold AUC stays within ±1.0pp",
                ],
            },
            {
                "id": "feature_selection",
                "title": "Feature selection to reduce spurious signals",
                "status": "optional",
                "body": (
                    f"Remove features with high group correlation but low permutation importance "
                    f"in {model_name}. Use SHAP rank delta to identify features that contribute "
                    "noise rather than signal differentially across groups."
                ),
                "success_criteria": [
                    "≥ 2 features removed",
                    "AUC unchanged (within ±0.5pp)",
                    "EO gap decreases by ≥ 10%",
                ],
            },
        ]
        return {
            "diagnosis": "Model complexity bias — overfitting group-specific noise",
            "summary": (
                f"Model complexity bias detected ({confidence * 100:.0f}% confidence). "
                f"EO gap {eo_gap * 100:.1f}pp on {model_name} may partly reflect overfitting "
                f"to group-specific patterns in {attr_label} rather than genuine predictive signal."
            ),
            "actions": actions,
            "safe_to_auto_fix": True,
            "warning": None,
        }


@app.post("/api/audit/stage/6")
def stage_6():
    """Guided remediation — deterministic conditional fix recommendations
    driven by Stage 5 root-cause diagnosis. Does NOT blindly apply fixes;
    unsafe interventions are explicitly blocked."""
    session_id = request.form.get("session_id") or request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id (run Stage 2 first)"}), 400
    sess = _get_session(session_id)
    if sess is None:
        return jsonify({"error": "session expired — please re-run Stage 2"}), 404

    primary_root_cause = (request.form.get("primary_root_cause") or "").strip()
    if not primary_root_cause:
        return jsonify({"error": "missing primary_root_cause (run Stage 5 first)"}), 400

    def _fv(field: str, default: float = 0.0) -> float:
        try:
            return float(request.form.get(field) or default)
        except (TypeError, ValueError):
            return default

    confidence      = _fv("confidence", 0.5)
    eo_gap          = _fv("eo_gap", 0.0)
    dp_gap          = _fv("dp_gap", 0.0)
    flip_rate       = _fv("flip_rate", 0.0)
    proxy_count     = int(_fv("proxy_count", 0))
    model_key       = (request.form.get("model_key") or "").strip() or None
    model_name      = (request.form.get("model_name") or model_key or "the model").strip()
    top_proxy       = (request.form.get("top_proxy_feature") or "").strip() or None
    protected_attr  = (request.form.get("protected_attr") or "").strip() or None

    # Frontend-supplied human-readable group specifics — passed through the
    # encoding decoder (lib/labels.ts) so we can name the actual minority group.
    minority_label = (request.form.get("minority_label") or "").strip() or None
    majority_label = (request.form.get("majority_label") or "").strip() or None
    minority_n_raw = request.form.get("minority_n")
    majority_n_raw = request.form.get("majority_n")
    try:
        minority_n = int(minority_n_raw) if minority_n_raw else None
    except (TypeError, ValueError):
        minority_n = None
    try:
        majority_n = int(majority_n_raw) if majority_n_raw else None
    except (TypeError, ValueError):
        majority_n = None
    attr_human_label = (request.form.get("attr_human_label") or "").strip() or None

    plan = _remediation_plan(
        primary_root_cause=primary_root_cause,
        confidence=confidence,
        eo_gap=eo_gap,
        dp_gap=dp_gap,
        flip_rate=flip_rate,
        proxy_count=proxy_count,
        top_proxy_feature=top_proxy,
        model_name=model_name,
        protected_attr=protected_attr,
        minority_label=minority_label,
        majority_label=majority_label,
        minority_n=minority_n,
        majority_n=majority_n,
        attr_human_label=attr_human_label,
    )

    return jsonify({
        "session_id": session_id,
        "model_key": model_key,
        "model_name": model_name,
        "primary_root_cause": primary_root_cause,
        **plan,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
