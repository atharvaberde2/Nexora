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
import os
import time
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError
from scipy.stats import chi2, chi2_contingency, norm

# Load backend/.env so GEMINI_API_KEY etc. are picked up at startup.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

# Gemini is optional — Stage 7 and 8 fall back to deterministic templates
# when either the SDK isn't installed or no API key is configured.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        HAS_GEMINI = True
    else:
        HAS_GEMINI = False
except Exception:
    genai = None  # type: ignore[assignment]
    HAS_GEMINI = False


def _gemini_narrate(
    prompt: str,
    system: str | None = None,
    max_chars: int = 4000,
    max_output_tokens: int = 2048,
) -> str | None:
    """Call Gemini for a short prose narrative. Returns None on any failure
    (missing key, network, quota, content filter) so callers can fall back to
    deterministic templates without breaking the audit.

    `max_output_tokens` defaults to 2048 because gemini-2.5-flash uses internal
    "thinking" tokens that count against this budget — a smaller cap gets
    consumed by thinking and produces empty visible output."""
    if not HAS_GEMINI or genai is None:
        return None
    try:
        kwargs = {}
        if system:
            kwargs["system_instruction"] = system
        model = genai.GenerativeModel(GEMINI_MODEL, **kwargs)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json"
                if "JSON" in (prompt or "") else "text/plain",
            },
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return None
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        return text
    except Exception:
        return None
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


def _clean_group_label(v) -> str | None:
    """Normalize a single protected-attribute value to a clean string label,
    or None if it's missing / NaN / a null sentinel.

    `protected_values` is supposed to come out of prepare_features as strings,
    but numeric-encoded categorical columns and pandas NaN floats can sneak
    through (e.g. when an upstream caller bypasses the str conversion). Keep
    this guard cheap and call it everywhere we iterate over group values."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


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
    # Track preprocessing decisions explicitly so the audit is transparent
    # about how many rows the user actually got group statistics for.
    n_input = int(len(df))
    n_target_missing = int(df[target_col].isna().sum())
    n_protected_missing = int(df[protected_col].isna().sum())
    df = df.dropna(subset=[target_col, protected_col]).copy()
    n_audited = int(len(df))
    n_dropped = n_input - n_audited

    if n_audited == 0:
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

    # Preprocessing transparency block — tells the user exactly what got
    # dropped before subgroup statistics were computed.
    preprocessing = {
        "n_input": n_input,
        "n_audited": n_audited,
        "n_dropped_total": n_dropped,
        "n_dropped_target_missing": n_target_missing,
        "n_dropped_protected_missing": n_protected_missing,
        "drop_rate": _safe_float(n_dropped / n_input) if n_input else None,
    }

    return {
        "n_total": n_audited,
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
        "preprocessing": preprocessing,
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
            {s for v in values if (s := _clean_group_label(v)) is not None}
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
            {s for v in values if (s := _clean_group_label(v)) is not None}
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
            {s for v in values if (s := _clean_group_label(v)) is not None}
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
            {s for v in values if (s := _clean_group_label(v)) is not None}
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
            # Degeneracy detection: a model whose selection rate is 0 across
            # every group (all-negative) or 1 across every group (all-positive)
            # has trivially zero TPR/FPR/EO/DP gaps — it looks "perfectly fair"
            # but is decision-theoretically useless. Without this flag, such a
            # model can survive the Pareto + fairness-threshold gates and get
            # recommended over a useful-but-imperfect classifier.
            srs = [g.get("selection_rate") for g in by_group.values()
                   if g.get("selection_rate") is not None]
            degenerate_kind: str | None = None
            if len(srs) >= 2:
                if all(s == 0 for s in srs):
                    degenerate_kind = "all-negative"
                elif all(s == 1 for s in srs):
                    degenerate_kind = "all-positive"
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
                "degenerate": degenerate_kind,
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

        # Step 3 — Selection rule: highest AUC among models that are
        #   (a) Pareto-optimal,
        #   (b) satisfy the fairness threshold,
        #   (c) are NOT degenerate (don't predict the same class for everyone).
        # Without (c), an all-zero classifier would have EO gap = 0 (trivially
        # fair) and slip through every prior filter despite being useless.
        # If no model qualifies, no recommendation is issued and a warning
        # surfaces so the user knows why.
        eligible = [
            r for r in rows
            if r["pareto_optimal"]
            and r["fairness_qualified"]
            and r["auc"] is not None
            and r.get("degenerate") is None
        ]
        recommendation_warning = None
        if eligible:
            best = max(eligible, key=lambda r: r["auc"])
            for r in rows:
                r["recommended"] = r is best
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
            degenerate_rows = [r for r in rows if r.get("degenerate") is not None]
            if pareto_rows and all(r.get("degenerate") for r in pareto_rows if r["fairness_qualified"]):
                # All "fairness-qualified" Pareto models are actually degenerate
                # (trivially fair because they predict one class for everyone).
                names = ", ".join(
                    f"{r['name']} ({r['degenerate']})" for r in degenerate_rows
                )
                recommendation_warning = (
                    "BLOCKED: every Pareto-optimal model that satisfies the fairness threshold "
                    f"is degenerate ({names}) — they predict a single class for every input, "
                    "so their gaps are trivially zero. Stage 4 refuses to recommend a useless model. "
                    "Inspect the non-degenerate candidates manually."
                )
            elif pareto_rows:
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


# =========================================================
# Stage 5 — real Bayesian root-cause inference
# =========================================================
#
# Generative model:
#   C ~ Categorical(π) over 5 root-cause classes
#   Z = (proxy_norm, flip_rate, imbalance_norm, base_rate_gap, eo_gap, dp_minus_eo)
#   Z_j | C=k  ~  Normal(μ_{k,j}, σ_j)
#
# Inference is exact: discrete C with a Gaussian likelihood factorized over
# observables, posterior P(C | Z) ∝ ∏_j P(Z_j | C) · π(C).
#
# Sampling: Monte Carlo over observable uncertainty (bootstrap-style noise on
# Z) gives a posterior distribution over P(C|Z) — we report its mean and
# 95% credible interval per class. This is real Bayesian inference, not a
# heuristic softmax of metrics.

# Domain-informed prior. Reflects how often each cause is reported as PRIMARY
# in the fairness-ML literature (Mehrabi et al. 2021, Suresh & Guttag 2021,
# Barocas-Hardt-Narayanan). Representation and proxy issues dominate; threshold
# and complexity bias tend to compound the others rather than be primary.
_BAYES_PRIOR = {
    "proxy_discrimination":  0.25,
    "representation_bias":   0.30,
    "label_bias":            0.25,
    "threshold_effect":      0.10,
    "model_complexity_bias": 0.10,
}
_BAYES_CLASSES = list(_BAYES_PRIOR.keys())

# Class-conditional means μ_{k,j} for the 6 standardized observables under each
# cause class. Hand-calibrated from the literature; adjust if domain shifts.
# Columns: [proxy_norm, flip_rate, imbalance_norm, base_rate_gap, eo_gap, dp_minus_eo]
_BAYES_MU = {
    # Proxy: high proxy ratio, high flip rate, moderate gaps, dp ≈ eo
    "proxy_discrimination":  [0.85, 0.50, 0.30, 0.10, 0.20, 0.05],
    # Representation: dominant imbalance, low flip, mild gaps, dp ≈ eo
    "representation_bias":   [0.20, 0.15, 0.85, 0.30, 0.20, 0.05],
    # Label: high base-rate gap, low flip rate, gaps follow base rate
    "label_bias":            [0.20, 0.05, 0.30, 0.85, 0.15, 0.05],
    # Threshold: dp_gap distinctly larger than eo_gap, moderate everywhere else.
    # μ for dp_minus_eo is 0.30 (a realistic strong signal — anything ≥0.15
    # is meaningful), not the extreme 0.85 it was originally calibrated to.
    "threshold_effect":      [0.20, 0.10, 0.20, 0.10, 0.20, 0.30],
    # Complexity: moderate eo, no specific signature on dp-eo gap.
    "model_complexity_bias": [0.40, 0.30, 0.30, 0.20, 0.45, 0.10],
}
# Per-observable measurement noise σ_j. dp_minus_eo gets a tighter σ because
# it's the only feature that uniquely distinguishes threshold effect from
# the others — small differences should be informative, not drowned out.
_BAYES_SIGMA = [0.20, 0.20, 0.20, 0.20, 0.20, 0.12]
# Bootstrap noise added to observables during MC sampling — captures the
# uncertainty in the point estimates we feed into Bayes' rule.
_BAYES_OBS_NOISE = 0.05
_BAYES_N_SAMPLES = 2000


def _bayes_standardize(
    proxy_score: float, flip_rate: float, group_imbalance: float,
    base_rate_gap: float, eo_gap: float, dp_gap: float,
) -> "np.ndarray":
    """Map raw observables to [0, 1] features for the Gaussian likelihood."""
    proxy_norm = float(np.clip((proxy_score - 1.0) / 4.0, 0.0, 1.0))   # ratio 1–5x → 0–1
    imbalance_norm = float(np.clip((group_imbalance - 1.0) / 9.0, 0.0, 1.0))  # 1–10x → 0–1
    return np.array([
        proxy_norm,
        float(np.clip(flip_rate, 0.0, 1.0)),
        imbalance_norm,
        float(np.clip(base_rate_gap, 0.0, 1.0)),
        float(np.clip(eo_gap, 0.0, 1.0)),
        float(np.clip(max(dp_gap - eo_gap, 0.0), 0.0, 1.0)),  # the threshold-effect signature
    ])


def _bayesian_posterior(
    proxy_score: float, flip_rate: float, group_imbalance: float,
    base_rate_gap: float, eo_gap: float, dp_gap: float,
) -> dict:
    """Backward-compatible wrapper: returns just posterior means per class.
    Internally computes the full Bayesian posterior with credible intervals
    via :func:`_bayesian_posterior_full`, then drops the CI envelope."""
    full = _bayesian_posterior_full(
        proxy_score, flip_rate, group_imbalance, base_rate_gap, eo_gap, dp_gap
    )
    return {cls: full[cls]["mean"] for cls in _BAYES_CLASSES}


def _bayesian_posterior_full(
    proxy_score: float, flip_rate: float, group_imbalance: float,
    base_rate_gap: float, eo_gap: float, dp_gap: float,
    *, n_samples: int = _BAYES_N_SAMPLES, seed: int = 42,
) -> dict:
    """Real Bayesian inference over 5 root-cause classes.

    Returns, per class, {mean, ci_low, ci_high} where the CIs are 2.5/97.5
    percentiles of the posterior under bootstrap-style observable uncertainty.
    """
    rng = np.random.default_rng(seed)
    Z = _bayes_standardize(proxy_score, flip_rate, group_imbalance,
                           base_rate_gap, eo_gap, dp_gap)
    log_prior = np.log(np.array([_BAYES_PRIOR[c] for c in _BAYES_CLASSES]))
    mu = np.array([_BAYES_MU[c] for c in _BAYES_CLASSES])  # (5, 6)
    sigma = np.array(_BAYES_SIGMA)                         # (6,)

    # MC posterior: for each draw, perturb observables with bootstrap-style
    # Gaussian noise, then compute exact discrete posterior given those Z.
    samples = np.zeros((n_samples, len(_BAYES_CLASSES)))
    for i in range(n_samples):
        Z_i = np.clip(Z + rng.normal(0.0, _BAYES_OBS_NOISE, size=Z.shape), 0.0, 1.0)
        # log P(Z_i | C=k) under independent Normal observables
        # = -0.5 Σ_j ((Z_ij - μ_kj) / σ_j)^2  (constants drop out post-norm)
        log_lik = -0.5 * np.sum(((Z_i - mu) / sigma) ** 2, axis=1)
        log_post = log_lik + log_prior
        log_post -= log_post.max()  # numerical stability
        post = np.exp(log_post)
        post /= post.sum()
        samples[i] = post

    mean = samples.mean(axis=0)
    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)
    return {
        cls: {
            "mean": _safe_float(mean[i]),
            "ci_low": _safe_float(ci_lo[i]),
            "ci_high": _safe_float(ci_hi[i]),
            "prior": _safe_float(_BAYES_PRIOR[cls]),
        }
        for i, cls in enumerate(_BAYES_CLASSES)
    }


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
            {s for v in values if (s := _clean_group_label(v)) is not None}
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

        posterior_full = _bayesian_posterior_full(
            float(proxy_score), float(fr_val), float(imbal),
            float(br_gap), float(eo_gap), float(dp_gap),
        )
        # Backward-compat dict (just means) used by Stages 6/7/8 that haven't
        # been migrated to consume credible intervals yet.
        posterior = {cls: posterior_full[cls]["mean"] for cls in _BAYES_CLASSES}

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
            # Full Bayesian output: per-class {mean, ci_low, ci_high, prior}
            # produced by Monte Carlo over observable bootstrap noise (n=2000).
            "bayesian_root_cause_full": posterior_full,
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


# =========================================================
# STAGE 7 — Reasoning / validation layer (4 Pydantic-checked checkpoints)
# =========================================================
#
# Stage 7 is a verification firewall on top of Stages 1–6. It does NOT retrain
# or recompute fairness metrics. It cross-checks prior stages for internal
# consistency and gates the final recommendation. Pydantic enforces the schema
# of every checkpoint so a malformed pipeline output fails loudly here rather
# than silently leaking into the report.

class CP1BiasValidation(BaseModel):
    """CP1 — Bias fingerprint normalized + checked for internal consistency."""
    n_total: int
    n_groups: int
    largest_imbalance_ratio: float | None = None
    smallest_group_n: int | None = None
    chi2_significant: bool = False
    chi2_p_value: float | None = None
    base_rate_gap_pp: float | None = None
    missingness_disparity_pp: float | None = None
    severity: Literal["low", "medium", "high"]
    inconsistencies: list[str] = Field(default_factory=list)
    summary: str = ""


class CP2PerModel(BaseModel):
    model_key: str
    model_name: str
    cv_auc: float | None = None
    eo_gap: float | None = None
    subgroup_auc_variance: float | None = None
    verdict: Literal["confirmed", "rejected", "ambiguous", "insufficient_data"]


class CP2ModelHypotheses(BaseModel):
    """CP2 — Tests whether higher accuracy correlates with worse fairness."""
    hypothesis: str = (
        "Higher CV AUC correlates with larger equalized-odds gap "
        "(accuracy/fairness tension)."
    )
    correlation_auc_eo: float | None = None
    verdict_summary: Literal["confirmed", "rejected", "ambiguous", "insufficient_data"]
    notes: list[str] = Field(default_factory=list)
    per_model: list[CP2PerModel]


class CP3RootCause(BaseModel):
    """CP3 — Cross-validates Stage 5's ML root cause vs Stage 1+3 statistical evidence."""
    statistical_root_cause: str
    statistical_evidence: list[str] = Field(default_factory=list)
    ml_inferred_root_cause: str
    agree: bool
    disagreement_flag: bool
    notes: list[str] = Field(default_factory=list)


class CP4FinalRecommendation(BaseModel):
    """CP4 — Gates the final pick: non-dominated + fairness-compliant only."""
    model: str | None = None
    model_key: str | None = None
    reason: str
    pareto_status: Literal["non-dominated", "no_recommendation"]
    fairness_compliant: bool
    eo_gap: float | None = None
    eo_gap_threshold: float
    auc: float | None = None


class DataRemediationAction(BaseModel):
    """A single data-level fix surfaced when no model passes CP4. Each one
    must explicitly explain why model-level fixes cannot reach the problem —
    that's the whole point of this branch."""
    id: str
    title: str
    body: str
    why_model_fix_insufficient: str
    success_criteria: list[str] = Field(default_factory=list)


class DataRemediation(BaseModel):
    """Dataset-level remediation surfaced ONLY when CP4.pareto_status is
    'no_recommendation' — i.e. every candidate model was blocked by Stage 4
    or Stage 7. Maps Stage 5 / CP3 root cause to data-cleaning steps."""
    triggered: bool
    headline: str = ""  # "No safe model exists under current data conditions"
    root_cause: str = "unknown"
    rationale: str = ""
    actions: list[DataRemediationAction] = Field(default_factory=list)


class Stage7Narratives(BaseModel):
    """LLM-generated plain-English summaries layered over the deterministic
    checkpoint output. The verdict (pass/fail) is NEVER taken from the LLM —
    only the prose. If Gemini isn't configured, every field falls back to
    the deterministic template equivalent."""
    cp1: str
    cp2: str
    cp3: str
    cp4: str
    executive_narrative: str
    llm_provider: Literal["gemini", "deterministic"]
    llm_model: str | None = None


class Stage7Response(BaseModel):
    session_id: str
    bias_validation: CP1BiasValidation
    model_hypotheses: CP2ModelHypotheses
    root_cause_consistency: CP3RootCause
    final_recommendation: CP4FinalRecommendation
    data_remediation: DataRemediation
    all_checkpoints_passed: bool
    checkpoints_summary: str
    narratives: Stage7Narratives


def _cp1_bias_validation(stage1: dict) -> CP1BiasValidation:
    """Read Stage 1 fingerprint, normalize, and flag inconsistencies."""
    results = (stage1 or {}).get("results") or []
    # Pick the attribute with the most groups for the headline numbers; the
    # report tab will show all attributes separately.
    primary = max(results, key=lambda r: len(r.get("fingerprint", {}).get("groups", []))) \
        if results else {"fingerprint": {}}
    fp = primary.get("fingerprint", {}) or {}
    groups = fp.get("groups") or []
    ns = [g.get("n", 0) for g in groups if g.get("n") is not None]
    pos_rates = [g.get("positive_rate") for g in groups if g.get("positive_rate") is not None]
    miss_rates = [g.get("missing_rate") for g in groups if g.get("missing_rate") is not None]

    largest_ratio = (max(ns) / max(min(ns), 1)) if ns else None
    base_rate_gap = (max(pos_rates) - min(pos_rates)) if len(pos_rates) >= 2 else None
    missing_disp = (max(miss_rates) - min(miss_rates)) if len(miss_rates) >= 2 else None
    smallest_n = min(ns) if ns else None
    lb = fp.get("label_bias") or {}
    chi2_p = lb.get("p_value")
    chi2_sig = bool(lb.get("significant"))

    inconsistencies: list[str] = []
    if chi2_sig and base_rate_gap is not None and base_rate_gap < 0.05:
        inconsistencies.append(
            f"Chi-square is significant (p={chi2_p:.3f}) but the base-rate gap "
            f"is small ({base_rate_gap * 100:.1f}pp) — significance may be "
            "driven by sample size rather than effect size."
        )
    if missing_disp is not None and missing_disp > 0.10 and not chi2_sig:
        inconsistencies.append(
            f"Group-correlated missingness ({missing_disp * 100:.1f}pp) without a "
            "significant label-bias signal — possible MNAR pattern worth manual review."
        )
    if largest_ratio is not None and largest_ratio > 5.0:
        inconsistencies.append(
            f"Severe group imbalance ({largest_ratio:.1f}× ratio between largest "
            "and smallest groups) — subgroup statistics for the minority will be unreliable."
        )
    if smallest_n is not None and smallest_n < 50:
        inconsistencies.append(
            f"Smallest group has only {smallest_n} samples — bootstrap CIs and "
            "subgroup AUC will be wide; treat point estimates with caution."
        )

    # Severity heuristic: blend imbalance, missingness, base-rate gap.
    severity_score = 0
    if largest_ratio is not None and largest_ratio > 3: severity_score += 1
    if largest_ratio is not None and largest_ratio > 6: severity_score += 1
    if base_rate_gap is not None and base_rate_gap > 0.10: severity_score += 1
    if missing_disp is not None and missing_disp > 0.10: severity_score += 1
    severity: Literal["low", "medium", "high"] = (
        "high" if severity_score >= 3 else "medium" if severity_score >= 1 else "low"
    )

    summary = (
        f"{len(ns)} groups, n={sum(ns):,}; "
        f"largest imbalance {largest_ratio:.1f}× " if largest_ratio else ""
    ) + (
        f"base-rate gap {base_rate_gap * 100:.1f}pp; " if base_rate_gap is not None else ""
    ) + (
        f"chi² p={chi2_p:.3f}" if chi2_p is not None else "chi² unavailable"
    )

    return CP1BiasValidation(
        n_total=int(sum(ns)),
        n_groups=len(ns),
        largest_imbalance_ratio=_safe_float(largest_ratio) if largest_ratio else None,
        smallest_group_n=smallest_n,
        chi2_significant=chi2_sig,
        chi2_p_value=_safe_float(chi2_p) if chi2_p is not None else None,
        base_rate_gap_pp=_safe_float(base_rate_gap * 100) if base_rate_gap is not None else None,
        missingness_disparity_pp=_safe_float(missing_disp * 100) if missing_disp is not None else None,
        severity=severity,
        inconsistencies=inconsistencies,
        summary=summary,
    )


def _cp2_model_hypotheses(stage2: dict, stage3: dict) -> CP2ModelHypotheses:
    """Test the canonical fairness hypothesis: does higher AUC come with larger EO gaps?"""
    s2_models = {m["key"]: m for m in (stage2.get("models") or [])}
    s3_results = stage3.get("results") or []
    if not s3_results:
        return CP2ModelHypotheses(
            verdict_summary="insufficient_data",
            per_model=[],
            notes=["Stage 3 results missing — cannot evaluate hypothesis."],
        )

    # Use the first protected attribute's per-model row.
    s3_models = s3_results[0].get("models") or []
    pairs: list[tuple[float, float, dict]] = []  # (auc, eo_gap, model_meta)
    per_model: list[CP2PerModel] = []
    for m in s3_models:
        eo = (m.get("gaps") or {}).get("eo_gap")
        auc = m.get("overall_auc")
        # Subgroup AUC variance — scaled fairness signal.
        bg = m.get("by_group") or {}
        sg_aucs = [g.get("auc") for g in bg.values() if g.get("auc") is not None]
        sg_var = float(np.var(sg_aucs)) if len(sg_aucs) >= 2 else None
        if auc is not None and eo is not None:
            pairs.append((float(auc), float(eo), m))

        per_model.append(CP2PerModel(
            model_key=m.get("key", ""),
            model_name=m.get("name", m.get("key", "")),
            cv_auc=_safe_float(s2_models.get(m.get("key"), {}).get("best_score") or auc),
            eo_gap=_safe_float(eo) if eo is not None else None,
            subgroup_auc_variance=_safe_float(sg_var) if sg_var is not None else None,
            verdict="insufficient_data",  # filled in below relative to median
        ))

    if len(pairs) < 3:
        return CP2ModelHypotheses(
            verdict_summary="insufficient_data",
            per_model=per_model,
            notes=[f"Only {len(pairs)} models have both AUC and EO gap — need ≥3 to test correlation."],
        )

    aucs = np.array([p[0] for p in pairs])
    eos = np.array([p[1] for p in pairs])
    # Pearson correlation (ddof handled implicitly by np.corrcoef).
    if aucs.std() < 1e-9 or eos.std() < 1e-9:
        r = None
    else:
        r = float(np.corrcoef(aucs, eos)[0, 1])

    if r is None:
        verdict = "insufficient_data"
    elif r > 0.5:
        verdict = "confirmed"
    elif r < -0.2:
        verdict = "rejected"
    else:
        verdict = "ambiguous"

    # Per-model: compare each to the median; "confirmed" if its AUC and EO gap
    # are both above (or both below) median (i.e., trend-consistent).
    auc_med = float(np.median(aucs))
    eo_med = float(np.median(eos))
    keyed = {m.get("key"): (a, e) for (a, e, m) in pairs}
    for entry in per_model:
        ae = keyed.get(entry.model_key)
        if ae is None:
            entry.verdict = "insufficient_data"
            continue
        a, e = ae
        if (a > auc_med and e > eo_med) or (a < auc_med and e < eo_med):
            entry.verdict = "confirmed"
        elif (a > auc_med and e < eo_med) or (a < auc_med and e > eo_med):
            entry.verdict = "rejected"
        else:
            entry.verdict = "ambiguous"

    notes = [
        f"Pearson correlation between AUC and EO gap across {len(pairs)} models: "
        f"r = {r:.3f}." if r is not None else "Variance too small to compute correlation."
    ]
    if verdict == "confirmed":
        notes.append("Higher-AUC models tend to be less fair — accept-fairness-as-constraint design is justified.")
    elif verdict == "rejected":
        notes.append("Higher-AUC models are also fairer here — fairness/accuracy may not be in tension on this dataset.")

    return CP2ModelHypotheses(
        correlation_auc_eo=_safe_float(r) if r is not None else None,
        verdict_summary=verdict,
        per_model=per_model,
        notes=notes,
    )


def _cp3_root_cause(stage1: dict, stage3: dict, stage5: dict) -> CP3RootCause:
    """Cross-validate Stage 5's diagnosis against statistical evidence from Stages 1 & 3."""
    s1_results = stage1.get("results") or []
    fp = s1_results[0].get("fingerprint") if s1_results else {}
    groups = (fp or {}).get("groups") or []
    ns = [g.get("n", 0) for g in groups]
    pos_rates = [g.get("positive_rate") for g in groups if g.get("positive_rate") is not None]
    largest_ratio = (max(ns) / max(min(ns), 1)) if ns else None
    base_rate_gap = (max(pos_rates) - min(pos_rates)) if len(pos_rates) >= 2 else None

    # Statistical-only inference: pick the cause that the data alone supports.
    evidence: list[str] = []
    if largest_ratio is not None and largest_ratio > 5.0:
        stat_cause = "representation_bias"
        evidence.append(f"Group-size ratio {largest_ratio:.1f}× exceeds 5× threshold.")
    elif base_rate_gap is not None and base_rate_gap > 0.15:
        stat_cause = "label_bias"
        evidence.append(f"Base-rate gap {base_rate_gap * 100:.1f}pp suggests label generation differs across groups.")
    elif largest_ratio is not None and largest_ratio > 2.0:
        stat_cause = "representation_bias"
        evidence.append(f"Moderate imbalance {largest_ratio:.1f}× — representation bias plausible.")
    else:
        # Look at Stage 3's DP vs EO gap to distinguish threshold from proxy.
        s3_models = (stage3.get("results") or [{}])[0].get("models") or []
        # Find recommended-ish model = highest AUC among any.
        scored = [m for m in s3_models if m.get("overall_auc") is not None]
        if scored:
            best = max(scored, key=lambda m: m["overall_auc"])
            dp = (best.get("gaps") or {}).get("dp_gap") or 0
            eo = (best.get("gaps") or {}).get("eo_gap") or 0
            if dp > eo + 0.05:
                stat_cause = "threshold_effect"
                evidence.append(f"DP gap ({dp * 100:.1f}pp) exceeds EO gap ({eo * 100:.1f}pp) by >5pp on best model.")
            else:
                stat_cause = "proxy_discrimination"
                evidence.append("EO and DP gaps are comparable — disparity persists at score level (proxy-like).")
        else:
            stat_cause = "unknown"
            evidence.append("Insufficient Stage 3 data to distinguish threshold from proxy.")

    ml_cause = stage5.get("primary_root_cause") or "unknown"
    agree = stat_cause == ml_cause
    disagreement_flag = not agree and stat_cause != "unknown" and ml_cause != "unknown"

    notes: list[str] = []
    if disagreement_flag:
        notes.append(
            f"Statistical evidence points to '{stat_cause}' while SHAP-based "
            f"diagnosis points to '{ml_cause}'. Both are reported; manual review "
            "recommended before applying remediation."
        )
    elif agree:
        notes.append(f"Statistical and ML-inferred diagnoses agree: {stat_cause}.")

    return CP3RootCause(
        statistical_root_cause=stat_cause,
        statistical_evidence=evidence,
        ml_inferred_root_cause=ml_cause,
        agree=agree,
        disagreement_flag=disagreement_flag,
        notes=notes,
    )


def _cp4_final_gate(stage4: dict) -> CP4FinalRecommendation:
    """Verify the recommended model is genuinely Pareto-optimal AND fairness-compliant."""
    threshold = stage4.get("eo_gap_threshold", EO_GAP_THRESHOLD)
    s4_results = stage4.get("results") or []
    if not s4_results:
        return CP4FinalRecommendation(
            reason="No Stage 4 results available — cannot verify recommendation.",
            pareto_status="no_recommendation",
            fairness_compliant=False,
            eo_gap_threshold=float(threshold),
        )

    # Use the first protected attribute (multi-attribute audits would loop).
    attr = s4_results[0]
    rec = next((m for m in (attr.get("models") or []) if m.get("recommended")), None)
    if rec is None:
        return CP4FinalRecommendation(
            reason=attr.get("recommendation_warning") or "No recommendation issued by Stage 4.",
            pareto_status="no_recommendation",
            fairness_compliant=False,
            eo_gap_threshold=float(threshold),
        )

    # Re-verify Pareto + fairness directly from the row to defend against
    # upstream tampering. CP4 is the firewall, not a passthrough.
    pareto_ok = bool(rec.get("pareto_optimal"))
    eo = rec.get("fairness_gap")
    fairness_ok = bool(rec.get("fairness_qualified")) and (
        eo is not None and eo <= float(threshold)
    )

    if not pareto_ok:
        reason = (
            f"BLOCKED: {rec.get('name')} is marked recommended but is NOT Pareto-optimal. "
            "Stage 7 refuses to ratify a dominated recommendation."
        )
        return CP4FinalRecommendation(
            model=rec.get("name"),
            model_key=rec.get("key"),
            reason=reason,
            pareto_status="no_recommendation",
            fairness_compliant=False,
            eo_gap=_safe_float(eo) if eo is not None else None,
            eo_gap_threshold=float(threshold),
            auc=_safe_float(rec.get("auc")) if rec.get("auc") is not None else None,
        )

    # Defense-in-depth: even if Stage 4 somehow recommended a degenerate model,
    # CP4 catches it here. Stage 4 already filters these out, but this gate
    # exists so a future Stage 4 refactor can't silently regress.
    if rec.get("degenerate"):
        kind = rec["degenerate"]
        reason = (
            f"BLOCKED: {rec.get('name')} predicts {kind} for every input — "
            "a degenerate classifier whose 'fair' gaps are trivially zero. "
            "Stage 7 refuses to ratify a useless recommendation."
        )
        return CP4FinalRecommendation(
            model=rec.get("name"),
            model_key=rec.get("key"),
            reason=reason,
            pareto_status="no_recommendation",
            fairness_compliant=False,
            eo_gap=_safe_float(eo) if eo is not None else None,
            eo_gap_threshold=float(threshold),
            auc=_safe_float(rec.get("auc")) if rec.get("auc") is not None else None,
        )

    if not fairness_ok:
        reason = (
            f"BLOCKED: {rec.get('name')} has EO gap "
            f"{(eo or 0) * 100:.1f}pp which exceeds the {threshold * 100:.0f}pp "
            "fairness threshold. Recommendation revoked."
        )
        return CP4FinalRecommendation(
            model=rec.get("name"),
            model_key=rec.get("key"),
            reason=reason,
            pareto_status="no_recommendation",
            fairness_compliant=False,
            eo_gap=_safe_float(eo) if eo is not None else None,
            eo_gap_threshold=float(threshold),
            auc=_safe_float(rec.get("auc")) if rec.get("auc") is not None else None,
        )

    reason = (
        f"{rec.get('name')} is non-dominated on the Pareto frontier with AUC "
        f"{(rec.get('auc') or 0):.3f} and EO gap {(eo or 0) * 100:.1f}pp "
        f"(within {threshold * 100:.0f}pp threshold)."
    )
    return CP4FinalRecommendation(
        model=rec.get("name"),
        model_key=rec.get("key"),
        reason=reason,
        pareto_status="non-dominated",
        fairness_compliant=True,
        eo_gap=_safe_float(eo) if eo is not None else None,
        eo_gap_threshold=float(threshold),
        auc=_safe_float(rec.get("auc")) if rec.get("auc") is not None else None,
    )


def _data_remediation_plan(
    cp3: "CP3RootCause",
    cp4: "CP4FinalRecommendation",
    stage1: dict,
) -> "DataRemediation":
    """Generate dataset-level remediation when CP4 has blocked every candidate.

    Returns triggered=False otherwise so Stage 8 can short-circuit display.
    The headline string is fixed by design ("No safe model exists under
    current data conditions") so downstream consumers can render it verbatim.
    """
    if cp4.pareto_status != "no_recommendation":
        return DataRemediation(
            triggered=False,
            root_cause=cp3.statistical_root_cause,
        )

    # Pull minority/majority from Stage 1 fingerprint to make recommendations specific.
    s1_results = stage1.get("results") or []
    primary = (
        max(s1_results, key=lambda r: len(r.get("fingerprint", {}).get("groups", [])))
        if s1_results
        else {"fingerprint": {}, "protected": "the protected attribute"}
    )
    fp = primary.get("fingerprint", {}) or {}
    groups = fp.get("groups") or []
    sized = sorted(groups, key=lambda g: g.get("n", 0)) if groups else []
    minority = sized[0] if sized else {}
    majority = sized[-1] if sized else {}
    minority_label = minority.get("name") or "the minority group"
    majority_label = majority.get("name") or "the majority group"
    minority_n = minority.get("n")
    majority_n = majority.get("n")
    attr = primary.get("protected") or "the protected attribute"

    # Pick the cause to act on. If statistical and ML diagnoses disagree, prefer
    # the more invasive / upstream cause — under-reacting is worse than over-reacting
    # when the gate has already blocked everything.
    cause = cp3.statistical_root_cause or "unknown"
    if cp3.disagreement_flag:
        priority = {
            "label_bias": 4,
            "proxy_discrimination": 3,
            "representation_bias": 2,
            "threshold_effect": 1,
            "unknown": 0,
        }
        cause = max(
            [cp3.statistical_root_cause, cp3.ml_inferred_root_cause],
            key=lambda c: priority.get(c or "unknown", 0),
        )

    headline = "No safe model exists under current data conditions"

    if cause == "representation_bias":
        ratio_text = ""
        if minority_n and majority_n:
            ratio_text = f" (currently {majority_n / max(minority_n, 1):.1f}× majority/minority sample ratio)"
        target_n = int((majority_n or 0) * 0.3) if majority_n else None
        actions = [
            DataRemediationAction(
                id="stratified_resampling",
                title=f"Stratified resampling for {minority_label}",
                body=(
                    f"Apply stratified oversampling (e.g., SMOTE on the training fold only) or "
                    f"inverse-frequency class weights so {minority_label} "
                    f"(n = {minority_n if minority_n is not None else 'small'}) contributes to "
                    f"decision-boundary learning at parity with {majority_label}."
                ),
                why_model_fix_insufficient=(
                    "Every candidate trained on the current sample inherits the same imbalance. "
                    "Post-hoc threshold tuning, in-processing fairness constraints, and reweighting "
                    "at inference time cannot synthesize the missing decision-boundary information "
                    f"from {minority_label} — the signal isn't in the trained weights to recover."
                ),
                success_criteria=[
                    f"Effective sample for {minority_label} reaches ≥ 30% of {majority_label} after weighting",
                    "EO gap on best-AUC model drops below the configured threshold on retraining",
                ],
            ),
            DataRemediationAction(
                id="targeted_collection",
                title=f"Targeted data collection for {minority_label}",
                body=(
                    f"Resampling synthesizes new points; it does not add new information. "
                    f"Collect additional real samples for {minority_label}{ratio_text} until the "
                    f"observed sample count is comparable to {majority_label}."
                ),
                why_model_fix_insufficient=(
                    f"Subgroup AUC and bootstrap confidence intervals for {minority_label} are unreliable "
                    "until n increases. No model selected on these unreliable estimates can be safely "
                    "deployed regardless of training method — the uncertainty is in the data, not the model."
                ),
                success_criteria=[
                    (
                        f"{minority_label} sample count reaches at least {target_n} (≥ 30% of {majority_label})"
                        if target_n
                        else f"{minority_label} sample count grows by ≥ 50% from current"
                    ),
                ],
            ),
        ]
        rationale = (
            f"Group imbalance on {attr} is the primary driver. {minority_label} is too small for any "
            "model to learn faithful decision boundaries, so every candidate was blocked at CP4. "
            "Algorithmic interventions (in-processing fairness constraints, post-processing threshold "
            "tuning, reweighting at inference) cannot manufacture the missing minority-class signal — "
            "only data-level changes can."
        )

    elif cause == "label_bias":
        actions = [
            DataRemediationAction(
                id="stratified_label_audit",
                title=f"Stratified label audit for {attr}",
                body=(
                    "Pull a stratified random sample (≥ 50 records per group) of training labels. "
                    f"Have domain experts re-label independently without access to {attr}. Compare "
                    "to the original labels and quantify per-group disagreement rate."
                ),
                why_model_fix_insufficient=(
                    "When the labels themselves encode the bias, any model trained on them inherits it. "
                    "SMOTE, reweighting, in-processing constraints, and threshold tuning all optimize "
                    "toward the biased target — they reduce the measurable gap while preserving the "
                    "underlying harm. Fairness metrics improve; outcomes do not."
                ),
                success_criteria=[
                    "≥ 50 labels per group reviewed",
                    "Inter-rater agreement κ ≥ 0.7 on the audit set",
                    "Per-group disagreement rate documented",
                ],
            ),
            DataRemediationAction(
                id="ground_truth_recalibration",
                title="Recalibrate ground-truth labels",
                body=(
                    "Where the audit shows systematic per-group label drift, recalibrate the affected "
                    "labels (correct, re-label, or remove disputed records) before any retraining round. "
                    "Document every change with reviewer ID and reason."
                ),
                why_model_fix_insufficient=(
                    "A 'fair' model trained on biased labels is statistical laundering — it produces "
                    "metrics that pass while the underlying decision pattern remains discriminatory. "
                    "Only fixing the labels removes the bias from the loss function the model is fitting."
                ),
                success_criteria=[
                    "Audit-trail of every label change preserved",
                    "Post-recalibration base-rate gap shrinks meaningfully",
                ],
            ),
        ]
        rationale = (
            f"Label bias is the diagnosed cause: training labels for {attr} carry historical decision "
            "patterns the model learned faithfully. No algorithmic intervention is appropriate while the "
            "target itself is biased — fairness metrics would improve, but the model would still encode "
            "the original discrimination. The fix is upstream of the model, in the labels."
        )

    elif cause == "proxy_discrimination":
        actions = [
            DataRemediationAction(
                id="proxy_feature_removal",
                title=f"Remove or orthogonalize proxy features for {attr}",
                body=(
                    f"Identify features statistically correlated with {attr} and remove them from the "
                    f"feature set, or orthogonalize them (residualize against {attr} on a held-out set). "
                    "Then re-run the audit on the cleaned feature schema."
                ),
                why_model_fix_insufficient=(
                    "When a proxy feature drives prediction, every model in the candidate set will learn "
                    "it — the proxy is in the feature space itself, not any one model. Per-group threshold "
                    "tuning masks the disparity without removing it and can introduce disparate-treatment "
                    "legal liability."
                ),
                success_criteria=[
                    "Counterfactual flip rate drops below 10% on retraining",
                    "Proxy SHAP-importance ratio falls below 2.0×",
                ],
            ),
            DataRemediationAction(
                id="schema_review",
                title="Domain-expert review of remaining features",
                body=(
                    "Have a domain expert review every feature still in the schema for legitimate "
                    f"predictive value independent of {attr}. Document each retained feature's causal "
                    "rationale before retraining."
                ),
                why_model_fix_insufficient=(
                    "Proxies are often subtle (zip code → race, occupation → gender). Automated "
                    "correlation screens catch the obvious ones; domain review catches the ones that "
                    "matter and that no model selection can fix."
                ),
                success_criteria=[
                    "Each retained feature has a documented causal rationale",
                    "Removed proxies do not eliminate legitimate predictive signal",
                ],
            ),
        ]
        rationale = (
            f"Proxy discrimination is the diagnosed cause: at least one feature is acting as an indirect "
            f"signal for {attr}. Because the proxy is encoded in the feature space, it is not specific to "
            "any single model — every candidate inherited it, which is why CP4 blocked them all. The fix "
            "is to clean the feature schema before retraining, not to tune the model."
        )

    elif cause == "threshold_effect":
        # Edge case: threshold effects are normally model-level. If we still got
        # 'no model safe', the diagnosis is probably masking a deeper cause.
        actions = [
            DataRemediationAction(
                id="manual_review",
                title="Escalate to manual review",
                body=(
                    "The diagnosed cause is threshold-effect, which is normally addressable via per-group "
                    "threshold optimization at the model level. The fact that CP4 still blocked every "
                    "candidate suggests an inconsistency upstream — investigate before applying any "
                    "data-level fix."
                ),
                why_model_fix_insufficient=(
                    "If threshold tuning were sufficient, at least one candidate should have passed CP4. "
                    "Either the diagnosis is incomplete (a deeper cause is masking as a threshold issue) "
                    "or the model search space was too narrow. Applying a data-level fix without that "
                    "diagnosis is premature."
                ),
                success_criteria=[
                    "Stage 5 root-cause confidence re-checked with expanded model search",
                    "If a deeper cause is confirmed, return to the corresponding remediation path",
                ],
            ),
        ]
        rationale = (
            "Threshold effects are normally model-level, so reaching 'no model safe' suggests the "
            "diagnosis may be incomplete. Manual review is required before applying invasive data changes."
        )

    else:  # unknown / model_complexity_bias
        actions = [
            DataRemediationAction(
                id="manual_review",
                title="Escalate to manual review",
                body=(
                    "Root cause could not be confidently diagnosed, yet every candidate model failed the "
                    "fairness gate. Bring in a domain expert to inspect the data fingerprint (Stage 1), "
                    "feature schema, and labeling process before any retraining or data modification."
                ),
                why_model_fix_insufficient=(
                    "Without a confirmed root cause, model-level fixes are guesses. Guessing on a "
                    "fairness-blocked dataset risks producing a model that *looks* fair while preserving "
                    "the underlying harm."
                ),
                success_criteria=[
                    "Confirmed root cause documented before next remediation cycle",
                ],
            ),
        ]
        rationale = (
            "Root cause is undetermined, so no data-cleaning recipe is safe to recommend automatically. "
            "Manual diagnosis must precede any intervention — model-level or data-level."
        )

    return DataRemediation(
        triggered=True,
        headline=headline,
        root_cause=cause,
        rationale=rationale,
        actions=actions,
    )


_S7_SYSTEM_PROMPT = (
    "You are a fairness-auditing reasoning layer. Given the deterministic "
    "checkpoint output of an ML fairness pipeline, write a short plain-English "
    "narrative for a non-technical reader. Hard rules: "
    "(1) Never contradict the numbers. "
    "(2) Never invent a verdict — use exactly the pass/fail given. "
    "(3) Never claim the model is fair if 'fairness_compliant' is false. "
    "(4) Be concise: 2–4 sentences. No bullet lists, no headings."
)


def _stage7_narratives(
    cp1: "CP1BiasValidation",
    cp2: "CP2ModelHypotheses",
    cp3: "CP3RootCause",
    cp4: "CP4FinalRecommendation",
    all_passed: bool,
) -> Stage7Narratives:
    """Build LLM narratives for each checkpoint. Falls back to deterministic
    templates if Gemini isn't configured."""

    def _fallback_cp1() -> str:
        if cp1.inconsistencies:
            return (
                f"Bias-fingerprint severity is {cp1.severity}. "
                f"{len(cp1.inconsistencies)} inconsistency flag"
                f"{'s' if len(cp1.inconsistencies) != 1 else ''} surfaced — review them before "
                "trusting downstream subgroup statistics."
            )
        return (
            f"Bias-fingerprint severity is {cp1.severity}; no contradictions "
            "between imbalance, missingness, and the chi-square label-bias signal."
        )

    def _fallback_cp2() -> str:
        if cp2.correlation_auc_eo is None:
            return (
                "Not enough valid models to test the accuracy/fairness tension hypothesis."
            )
        verdict_text = {
            "confirmed": "higher-AUC models tend to be less fair on this dataset",
            "rejected": "higher-AUC models are also fairer here — no tension observed",
            "ambiguous": "no clear correlation either way",
            "insufficient_data": "not enough data to decide",
        }.get(cp2.verdict_summary, cp2.verdict_summary)
        return (
            f"Pearson r between AUC and EO gap is {cp2.correlation_auc_eo:.3f} — "
            f"{verdict_text}. This is why Stage 4 enforces fairness as a constraint, "
            "not just a tiebreaker."
        )

    def _fallback_cp3() -> str:
        if cp3.disagreement_flag:
            return (
                f"Statistical evidence points to '{cp3.statistical_root_cause.replace('_', ' ')}' "
                f"while the SHAP-based diagnosis points to '{cp3.ml_inferred_root_cause.replace('_', ' ')}'. "
                "Both are reported; manual review is recommended before applying any remediation."
            )
        return (
            f"Statistical and ML-inferred diagnoses agree: "
            f"{cp3.statistical_root_cause.replace('_', ' ')}. The remediation engine can act on this with confidence."
        )

    def _fallback_cp4() -> str:
        return cp4.reason

    def _fallback_exec() -> str:
        if all_passed:
            return (
                "All four checkpoints passed. The recommended model is non-dominated, "
                "within the fairness threshold, and the diagnosis story is internally consistent. "
                "Stage 7 ratifies the recommendation."
            )
        return (
            "Stage 7 found gaps that need human review before deployment. "
            "See the failing checkpoints below for what to address."
        )

    if not HAS_GEMINI:
        return Stage7Narratives(
            cp1=_fallback_cp1(),
            cp2=_fallback_cp2(),
            cp3=_fallback_cp3(),
            cp4=_fallback_cp4(),
            executive_narrative=_fallback_exec(),
            llm_provider="deterministic",
            llm_model=None,
        )

    # Build a single prompt that emits all 5 narratives as JSON. Cheaper and
    # avoids 5 round trips. Falls back per-field if Gemini returns garbage.
    facts = {
        "cp1": cp1.model_dump(),
        "cp2": cp2.model_dump(),
        "cp3": cp3.model_dump(),
        "cp4": cp4.model_dump(),
        "all_checkpoints_passed": all_passed,
    }
    prompt = (
        "Below is the JSON output of a fairness-audit reasoning layer (4 checkpoints). "
        "Write a short plain-English narrative for each. Return ONE JSON object with "
        "exactly these keys: cp1, cp2, cp3, cp4, executive_narrative. Each value is a "
        "string, 2–4 sentences. No markdown, no prose outside the JSON.\n\n"
        f"DATA:\n{json.dumps(facts, default=str)}"
    )
    text = _gemini_narrate(prompt, system=_S7_SYSTEM_PROMPT, max_chars=4000)
    parsed: dict | None = None
    if text:
        # Strip any code fences Gemini may add.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").lstrip("json").strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        # Whole-of-LLM failure — fall back uniformly.
        return Stage7Narratives(
            cp1=_fallback_cp1(),
            cp2=_fallback_cp2(),
            cp3=_fallback_cp3(),
            cp4=_fallback_cp4(),
            executive_narrative=_fallback_exec(),
            llm_provider="deterministic",
            llm_model=None,
        )

    def _pick(key: str, fallback: str) -> str:
        v = parsed.get(key) if parsed else None
        return v.strip() if isinstance(v, str) and v.strip() else fallback

    return Stage7Narratives(
        cp1=_pick("cp1", _fallback_cp1()),
        cp2=_pick("cp2", _fallback_cp2()),
        cp3=_pick("cp3", _fallback_cp3()),
        cp4=_pick("cp4", _fallback_cp4()),
        executive_narrative=_pick("executive_narrative", _fallback_exec()),
        llm_provider="gemini",
        llm_model=GEMINI_MODEL,
    )


@app.post("/api/audit/stage/7")
def stage_7():
    """Reasoning + validation layer. Cross-checks Stages 1-6 via four
    Pydantic-validated checkpoints, then narrates them via Gemini (with
    deterministic fallback). Does NOT retrain or recompute fairness."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id"}), 400

    s1 = payload.get("stage1") or {}
    s2 = payload.get("stage2") or {}
    s3 = payload.get("stage3") or {}
    s4 = payload.get("stage4") or {}
    s5 = payload.get("stage5") or {}

    cp1 = _cp1_bias_validation(s1)
    cp2 = _cp2_model_hypotheses(s2, s3)
    cp3 = _cp3_root_cause(s1, s3, s5)
    cp4 = _cp4_final_gate(s4)
    data_remediation = _data_remediation_plan(cp3, cp4, s1)

    all_passed = (
        cp1.severity != "high"
        and cp2.verdict_summary in {"confirmed", "rejected"}
        and not cp3.disagreement_flag
        and cp4.fairness_compliant
    )

    summary_parts = [
        f"CP1 ({cp1.severity} severity, {len(cp1.inconsistencies)} flags)",
        f"CP2 ({cp2.verdict_summary})",
        f"CP3 ({'AGREE' if cp3.agree else 'DISAGREE'})",
        f"CP4 ({cp4.pareto_status})",
    ]
    summary = " · ".join(summary_parts)

    narratives = _stage7_narratives(cp1, cp2, cp3, cp4, all_passed)

    try:
        result = Stage7Response(
            session_id=session_id,
            bias_validation=cp1,
            model_hypotheses=cp2,
            root_cause_consistency=cp3,
            final_recommendation=cp4,
            data_remediation=data_remediation,
            all_checkpoints_passed=all_passed,
            checkpoints_summary=summary,
            narratives=narratives,
        )
    except ValidationError as e:
        return jsonify({"error": f"Stage 7 schema validation failed: {e}"}), 500

    return jsonify(result.model_dump())


# =========================================================
# STAGE 8 — Decision-intelligence report (5 tabs)
# =========================================================

def _stage8_executive(stage7: dict) -> dict:
    """TAB 1 — recommended model + business interpretation. Reads from Stage 7's
    CP4 final_recommendation block (which has already verified non-dominance
    and fairness compliance), not directly from Stage 4 — keeping the trust
    chain Stage 4 → Stage 7 → Stage 8.

    When CP4 returns no_recommendation, this tab pivots to the data-remediation
    headline ("No safe model exists under current data conditions") and surfaces
    the dataset-level fix list that Stage 7 produced."""
    rec = stage7.get("final_recommendation") or {}
    dr = stage7.get("data_remediation") or {}
    threshold = rec.get("eo_gap_threshold", EO_GAP_THRESHOLD)
    if not rec.get("model"):
        # Every candidate was blocked. Lead with the canonical headline and
        # surface the data-remediation list as the next-action block.
        headline = dr.get("headline") or "No safe model exists under current data conditions"
        rationale = dr.get("rationale") or (
            "No model in this audit cleared both the Pareto-optimality test and "
            "the fairness-threshold guardrail."
        )
        return {
            "model": None,
            "auc": None,
            "eo_gap": None,
            "status": "no_recommendation",
            "headline": headline,
            "reason": rec.get("reason", "No model met the selection criteria."),
            "business_interpretation": (
                f"{headline}. {rationale} Deployment is not recommended at this time; "
                "see the dataset-level remediation steps below before retraining."
            ),
            "data_remediation": dr if dr.get("triggered") else None,
        }
    return {
        "model": rec["model"],
        "auc": rec.get("auc"),
        "eo_gap": rec.get("eo_gap"),
        "status": rec["pareto_status"],
        "reason": rec["reason"],
        "business_interpretation": (
            f"This model improves prediction quality while maintaining equitable "
            f"outcomes within a {threshold * 100:.0f}pp equalized-odds gap, making it "
            "suitable for deployment in regulated decision systems (lending, insurance, "
            "healthcare). It has been verified by Stage 7 as non-dominated and "
            "fairness-compliant."
        ),
        "data_remediation": None,
    }


def _stage8_fairness_risk(stage1: dict, stage5: dict, stage7: dict) -> dict:
    """TAB 2 — disadvantaged groups + bias type + statistical evidence."""
    cp1 = stage7.get("bias_validation") or {}
    cp3 = stage7.get("root_cause_consistency") or {}

    s1_results = stage1.get("results") or []
    disadvantaged: list[dict] = []
    for entry in s1_results:
        attr = entry.get("protected", "")
        fp = entry.get("fingerprint") or {}
        groups = fp.get("groups") or []
        if len(groups) < 2:
            continue
        sized = sorted(groups, key=lambda g: g.get("n", 0))
        small, large = sized[0], sized[-1]
        ratio = (large.get("n", 0) / max(small.get("n", 1), 1))
        disadvantaged.append({
            "attribute": attr,
            "minority_group": small.get("name"),
            "minority_n": small.get("n"),
            "majority_group": large.get("name"),
            "majority_n": large.get("n"),
            "imbalance_ratio": _safe_float(ratio),
        })

    # Risk severity: bubble up CP1 severity, but escalate if CP3 disagrees.
    severity = cp1.get("severity", "low")
    if cp3.get("disagreement_flag"):
        severity = "high" if severity != "high" else severity

    return {
        "severity": severity,
        "primary_bias_type": (
            stage5.get("primary_root_cause")
            or cp3.get("statistical_root_cause")
            or "unknown"
        ),
        "statistical_root_cause": cp3.get("statistical_root_cause"),
        "ml_inferred_root_cause": cp3.get("ml_inferred_root_cause"),
        "diagnoses_agree": cp3.get("agree", True),
        "disadvantaged_groups": disadvantaged,
        "evidence": {
            "chi2_p_value": cp1.get("chi2_p_value"),
            "chi2_significant": cp1.get("chi2_significant"),
            "base_rate_gap_pp": cp1.get("base_rate_gap_pp"),
            "missingness_disparity_pp": cp1.get("missingness_disparity_pp"),
            "largest_imbalance_ratio": cp1.get("largest_imbalance_ratio"),
        },
        "inconsistencies": cp1.get("inconsistencies", []),
    }


def _stage8_model_behavior(stage5: dict) -> dict:
    """TAB 3 — plain-language SHAP + proxy explanation."""
    results = stage5.get("results") or {}
    first_attr = next(iter(results.values()), {}) if results else {}
    proxy_feats = first_attr.get("proxy_features") or []
    corr_feats = first_attr.get("correlated_features") or []
    group_shap = first_attr.get("group_shap") or {}

    # Top features by SHAP across all groups (union of top-3 from each group's importance).
    top_features: list[str] = []
    for g, gs in group_shap.items():
        imp = gs.get("importance", {}) or {}
        top_features.extend(list(imp.keys())[:3])
    # Dedup preserving order.
    seen = set()
    top_features = [f for f in top_features if not (f in seen or seen.add(f))][:5]

    return {
        "top_features": top_features,
        "proxy_features": [
            {"feature": p["feature"], "ratio": p.get("ratio")}
            for p in proxy_feats[:5]
        ],
        "correlated_features": [
            {"feature": c["feature"], "smd": c.get("smd")}
            for c in corr_feats[:5]
        ],
        "shap_available": stage5.get("shap_available", False),
        "narrative": _stage8_behavior_narrative(top_features, proxy_feats, stage5),
    }


def _stage8_behavior_narrative(top_features: list, proxy_feats: list, stage5: dict) -> str:
    """Plain-language paragraph synthesizing what the model is doing."""
    if not top_features:
        return "No SHAP attribution was available for the recommended model."
    feats_str = ", ".join(top_features[:3])
    base = f"The model relies most heavily on {feats_str} when making its decisions."
    if proxy_feats:
        top_proxy = proxy_feats[0]
        ratio = top_proxy.get("ratio") or 1.0
        base += (
            f" The feature \"{top_proxy['feature']}\" is "
            f"{ratio:.1f}× more influential for one group than the other — "
            "a sign it may be acting as an indirect signal for the protected attribute "
            "even though that attribute itself was excluded from training."
        )
    else:
        base += " No strong proxy features were detected — feature usage is consistent across groups."
    return base


def _stage8_actions(stage6: dict, stage7: dict) -> dict:
    """TAB 4 — actionable recommendations.

    When CP4 has recommended a model, surface Stage 6's model-level fix list.
    When CP4 has blocked every candidate (no_recommendation), pivot to Stage 7's
    data-level remediation list — the model-level actions are no longer the
    relevant next step. Whichever list is shown, `mode` tells the consumer."""
    cp4 = stage7.get("final_recommendation") or {}
    dr = stage7.get("data_remediation") or {}
    if dr.get("triggered"):
        # All-blocked path — Stage 6's model-level actions don't apply.
        data_actions = dr.get("actions") or []
        return {
            "mode": "data_remediation",
            "diagnosis": dr.get("root_cause"),
            "summary": dr.get("rationale"),
            "headline": dr.get("headline"),
            "safe_to_auto_fix": False,
            "warning": (
                "No model in this audit is safe to deploy. The actions below operate "
                "on the dataset, not on any single model — model-level fixes cannot "
                "reach the diagnosed root cause."
            ),
            "actions": data_actions,
            "blocked_count": 0,
            "recommended_count": len(data_actions),
            "verified_by_stage7": False,
        }

    actions = (stage6 or {}).get("actions") or []
    return {
        "mode": "model_remediation",
        "diagnosis": stage6.get("diagnosis"),
        "summary": stage6.get("summary"),
        "headline": None,
        "safe_to_auto_fix": stage6.get("safe_to_auto_fix", False),
        "warning": stage6.get("warning"),
        "actions": actions,
        "blocked_count": sum(1 for a in actions if a.get("status") == "blocked"),
        "recommended_count": sum(1 for a in actions if a.get("status") == "recommended"),
        "verified_by_stage7": cp4.get("fairness_compliant", False),
    }


def _stage8_deployment(stage7: dict) -> dict:
    """TAB 5 — final deployment verdict synthesizing all four checkpoints."""
    cp1 = stage7.get("bias_validation") or {}
    cp3 = stage7.get("root_cause_consistency") or {}
    cp4 = stage7.get("final_recommendation") or {}
    all_passed = stage7.get("all_checkpoints_passed", False)

    conditions: list[dict] = [
        {
            "name": "Fairness threshold satisfied",
            "passed": bool(cp4.get("fairness_compliant")),
            "detail": (
                f"EO gap {(cp4.get('eo_gap') or 0) * 100:.1f}pp vs "
                f"{(cp4.get('eo_gap_threshold') or 0) * 100:.0f}pp threshold"
                if cp4.get("eo_gap") is not None else "No recommendation to evaluate"
            ),
        },
        {
            "name": "Recommended model is non-dominated",
            "passed": cp4.get("pareto_status") == "non-dominated",
            "detail": cp4.get("reason", ""),
        },
        {
            "name": "Statistical and ML diagnoses agree",
            "passed": not cp3.get("disagreement_flag", False),
            "detail": (
                f"Statistical: {cp3.get('statistical_root_cause', 'n/a')} · "
                f"ML: {cp3.get('ml_inferred_root_cause', 'n/a')}"
            ),
        },
        {
            "name": "Bias-fingerprint severity acceptable",
            "passed": cp1.get("severity") != "high",
            "detail": f"Severity = {cp1.get('severity', 'unknown')}",
        },
    ]
    passed_count = sum(1 for c in conditions if c["passed"])

    if all_passed and passed_count == 4:
        verdict = "deploy"
        verdict_text = "Deploy — all four reasoning checkpoints passed."
    elif passed_count >= 3 and cp4.get("fairness_compliant"):
        # 3/4 with CP4 holding = one soft signal failed (Pareto, diagnosis
        # agreement, or fingerprint severity). Conditional is appropriate.
        # 2/4 used to land here too — but two simultaneous failures of those
        # three is genuinely concerning and now downgrades to do_not_deploy.
        verdict = "conditional"
        verdict_text = (
            f"Conditional deploy — {passed_count}/4 checkpoints passed. "
            "Address the failing condition before production rollout."
        )
    else:
        verdict = "do_not_deploy"
        verdict_text = (
            f"Do not deploy — only {passed_count}/4 checkpoints passed and the "
            "fairness or non-dominance gate is not satisfied."
        )

    return {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "passed_count": passed_count,
        "total_conditions": len(conditions),
        "conditions": conditions,
    }


_S8_SYSTEM_PROMPT = (
    "You are writing an executive fairness report for product managers and "
    "compliance reviewers. Hard rules: "
    "(1) Never invent numbers — only use what the JSON gives you. "
    "(2) Never claim the model is fair if 'fairness_compliant' is false. "
    "(3) Never recommend deployment if 'deployment.verdict' is not 'deploy'. "
    "(4) If 'executive.data_remediation' is present, every narrative MUST open with the "
    "exact sentence in 'executive.headline' (no paraphrase) and refer ONLY to dataset-level "
    "fixes — never recommend a model, never imply one is suitable. "
    "(5) Be concise: 2–4 sentences per field. No markdown, no headings, plain prose."
)


def _stage8_llm_narratives(report: dict) -> dict:
    """Generate executive narrative, business interpretation, and behavior
    narrative via Gemini. Returns a dict of overrides to merge into the
    deterministic report; missing keys mean 'keep the deterministic version'."""
    if not HAS_GEMINI:
        return {"llm_provider": "deterministic", "llm_model": None}

    # Slim payload — only the facts needed for narration. Avoids leaking SHAP
    # arrays into the prompt and keeps token usage low.
    exec_block = report["executive"]
    dr_summary = None
    if exec_block.get("data_remediation"):
        dr = exec_block["data_remediation"]
        dr_summary = {
            "headline": dr.get("headline"),
            "root_cause": dr.get("root_cause"),
            "rationale": dr.get("rationale"),
            "action_titles": [a.get("title") for a in (dr.get("actions") or [])],
        }
    facts = {
        "executive": {
            "model": exec_block.get("model"),
            "auc": exec_block.get("auc"),
            "eo_gap": exec_block.get("eo_gap"),
            "status": exec_block.get("status"),
            "headline": exec_block.get("headline"),
            "data_remediation": dr_summary,
        },
        "fairness_risk": {
            "severity": report["fairness_risk"].get("severity"),
            "primary_bias_type": report["fairness_risk"].get("primary_bias_type"),
            "diagnoses_agree": report["fairness_risk"].get("diagnoses_agree"),
            "disadvantaged_groups": report["fairness_risk"].get("disadvantaged_groups"),
        },
        "model_behavior": {
            "top_features": report["model_behavior"].get("top_features"),
            "proxy_features": report["model_behavior"].get("proxy_features"),
            "shap_available": report["model_behavior"].get("shap_available"),
        },
        "actions": {
            "diagnosis": report["actions"].get("diagnosis"),
            "recommended_count": report["actions"].get("recommended_count"),
            "blocked_count": report["actions"].get("blocked_count"),
            "safe_to_auto_fix": report["actions"].get("safe_to_auto_fix"),
        },
        "deployment": {
            "verdict": report["deployment"].get("verdict"),
            "passed_count": report["deployment"].get("passed_count"),
            "total_conditions": report["deployment"].get("total_conditions"),
        },
    }
    prompt = (
        "Below is a fairness-audit decision report. Generate ONE JSON object with these "
        "exact keys (each value a 2–4 sentence string):\n"
        "  executive_narrative — overall plain-English headline a CEO/PM could read in 30s.\n"
        "  business_interpretation — risk + suitability framing for regulated decision systems.\n"
        "  behavior_narrative — what the model relies on, including any proxy concern, in plain language.\n"
        "  deployment_rationale — why the verdict was reached, citing the failing/passing checkpoints.\n"
        "Return ONLY the JSON, no markdown, no prose outside it.\n\n"
        f"DATA:\n{json.dumps(facts, default=str)}"
    )
    text = _gemini_narrate(prompt, system=_S8_SYSTEM_PROMPT, max_chars=4000)
    if not text:
        return {"llm_provider": "deterministic", "llm_model": None}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip("json").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return {"llm_provider": "deterministic", "llm_model": None}

    return {
        "executive_narrative": parsed.get("executive_narrative") if isinstance(parsed.get("executive_narrative"), str) else None,
        "business_interpretation": parsed.get("business_interpretation") if isinstance(parsed.get("business_interpretation"), str) else None,
        "behavior_narrative": parsed.get("behavior_narrative") if isinstance(parsed.get("behavior_narrative"), str) else None,
        "deployment_rationale": parsed.get("deployment_rationale") if isinstance(parsed.get("deployment_rationale"), str) else None,
        "llm_provider": "gemini",
        "llm_model": GEMINI_MODEL,
    }


@app.post("/api/audit/stage/8")
def stage_8():
    """Decision-intelligence report layer. Aggregates all prior stages into
    five tabs: Executive · Fairness & Risk · Model Behavior · Actions · Deployment.
    LLM narratives (via Gemini) are layered on top of the deterministic numbers
    when an API key is configured; otherwise the deterministic templates ship."""
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "missing session_id"}), 400

    s1 = payload.get("stage1") or {}
    s4 = payload.get("stage4") or {}
    s5 = payload.get("stage5") or {}
    s6 = payload.get("stage6") or {}
    s7 = payload.get("stage7") or {}

    report = {
        "session_id": session_id,
        "executive": _stage8_executive(s7),
        "fairness_risk": _stage8_fairness_risk(s1, s5, s7),
        "model_behavior": _stage8_model_behavior(s5),
        "actions": _stage8_actions(s6, s7),
        "deployment": _stage8_deployment(s7),
    }

    # Layer Gemini narratives on top — these REPLACE the corresponding template
    # strings when present. Numbers and verdicts are untouched.
    llm = _stage8_llm_narratives(report)
    if llm.get("business_interpretation"):
        report["executive"]["business_interpretation"] = llm["business_interpretation"]
    if llm.get("behavior_narrative"):
        report["model_behavior"]["narrative"] = llm["behavior_narrative"]
    if llm.get("deployment_rationale"):
        report["deployment"]["verdict_text"] = llm["deployment_rationale"]
    report["executive_narrative"] = llm.get("executive_narrative")  # may be None
    report["llm_provider"] = llm.get("llm_provider", "deterministic")
    report["llm_model"] = llm.get("llm_model")

    return jsonify(report)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
