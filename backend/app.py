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
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV
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

# Quiet down Optuna's per-trial chatter — the Flask logs stay readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_TRIALS = 10
CV_FOLDS = 5
SCORING = "roc_auc"
BOOTSTRAP_N = 1000
N_THRESHOLD = 100  # Power threshold for stage 1 + small-group warning in 3

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
) -> tuple[np.ndarray, np.ndarray, int, dict[str, np.ndarray]]:
    """Encode target, drop target+protected from features, impute, one-hot.

    Also returns `protected_values` — a dict mapping each protected column
    name to a per-row str array aligned with the returned X/y, so Stage 3
    can split predictions by group without re-reading the CSV.
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

    return X.values.astype(float), y.astype(int), X.shape[1], protected_values


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


def _confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float, float]:
    """Returns (TPR, FPR, FNR, TNR, selection_rate)."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)
    fnr = fn / (fn + tp + EPS)
    tnr = tn / (tn + fp + EPS)
    selection_rate = (tp + fp) / max(len(y_true), 1)
    return tpr, fpr, fnr, tnr, selection_rate


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
            "selection_rate": None,
            "auc": None, "auc_ci": [None, None],
            "sufficient_power": False,
        }

    tpr, fpr, fnr, tnr, sr = _confusion_rates(y_true_g, y_pred_g)
    try:
        auc = float(roc_auc_score(y_true_g, y_proba_g)) if len(np.unique(y_true_g)) >= 2 else None
    except Exception:
        auc = None

    # Bootstrap CIs only for the metrics judges typically inspect.
    tpr_samples: list[float] = []
    fpr_samples: list[float] = []
    auc_samples: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yt = y_true_g[idx]
        yp = y_pred_g[idx]
        ypr = y_proba_g[idx]
        t, f, _, _, _ = _confusion_rates(yt, yp)
        tpr_samples.append(t)
        fpr_samples.append(f)
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
        }

    def _spread(field: str) -> float | None:
        xs = [m[field] for m in valid if m[field] is not None]
        return (max(xs) - min(xs)) if len(xs) >= 2 else None

    tpr_gap = _spread("tpr")
    fpr_gap = _spread("fpr")
    eo_components = [v for v in (tpr_gap, fpr_gap) if v is not None]
    eo_gap = max(eo_components) if eo_components else None
    dp_gap = _spread("selection_rate")
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
        X, y, n_features, protected_values = prepare_features(
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

        # Cache the audit so Stages 3 & 4 can reuse it without re-training.
        _store_session(session_id, {
            "X": X,
            "y": y,
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

    # Per-model overall AUC + bootstrap CI is independent of protected attr —
    # compute once.
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
        }

    results = []
    for col, values in protected_values.items():
        # Filter out NaN-like group labels — they shouldn't drive a fairness
        # comparison, and the chi-square in Stage 1 already excluded them.
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
        "results": results,
    })


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
                tpr, fpr, fnr, tnr, sr = _confusion_rates(y[mask], y_pred[mask])
                by_group[g] = {
                    "n": int(mask.sum()),
                    "tpr": _safe_float(tpr),
                    "fpr": _safe_float(fpr),
                    "fnr": _safe_float(fnr),
                    "tnr": _safe_float(tnr),
                    "selection_rate": _safe_float(sr),
                }
            gaps = _compute_gaps(by_group)
            rows.append({
                "key": key,
                "name": pred["name"],
                "family": pred["family"],
                "color": pred["color"],
                "auc": _safe_float(overall_auc[key]),
                "fairness_gap": gaps["eo_gap"],
                "tpr_gap": gaps["tpr_gap"],
                "fpr_gap": gaps["fpr_gap"],
                "dp_gap": gaps["dp_gap"],
                "di_ratio": gaps["di_ratio"],
            })

        # Pareto dominance: r is dominated if some other row has auc>=r.auc
        # AND fairness_gap<=r.fairness_gap, with at least one strict.
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

        # Recommended = highest AUC among Pareto-optimal rows.
        pareto_rows = [r for r in rows if r["pareto_optimal"]]
        if pareto_rows:
            best = max(pareto_rows, key=lambda r: r["auc"])
            for r in rows:
                r["recommended"] = r is best
        else:
            for r in rows:
                r["recommended"] = False

        results.append({
            "protected": col,
            "models": rows,
        })

    return jsonify({"session_id": session_id, "results": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
