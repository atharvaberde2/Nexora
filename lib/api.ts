import type { ColumnType, ParsedCsv } from "./csv";
import type { AuditConfig } from "@/components/audit/stage-configure";

const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_URL) ||
  "http://localhost:5000";

/* ─────────────  CSV inspect (pandas-authoritative types)  ───────────── */

export type InspectColumn = {
  name: string;
  type: ColumnType;
  unique: number;
  missing_pct: number;
  sample: string[];
  unique_values: string[] | null;
  numeric_range: [number, number] | null;
};

export type InspectResponse = {
  columns: InspectColumn[];
  row_count: number;
};

export async function inspectCsv(
  csv: ParsedCsv,
  signal?: AbortSignal
): Promise<InspectResponse> {
  const fd = new FormData();
  const blob = new Blob([csv.text], { type: "text/csv" });
  fd.append("file", blob, csv.fileName);

  const res = await fetch(`${API_BASE}/api/csv/inspect`, {
    method: "POST",
    body: fd,
    signal,
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Inspect"));
  }
  return res.json();
}

/* ─────────────  Stage 1 (multi-protected)  ───────────── */

export type Stage1Group = {
  name: string;
  n: number;
  positive_rate: number | null;
  base_rate_gap: number | null;
  missing_rate: number | null;
  sufficient_power: boolean;
};

export type Stage1Fingerprint = {
  n_total: number;
  overall_positive_rate: number | null;
  n_threshold: number;
  groups: Stage1Group[];
  label_bias: {
    chi2: number | null;
    p_value: number | null;
    significant: boolean;
  };
  top_missing_columns: { name: string; missing_pct: number | null }[];
};

export type Stage1PerAttr = {
  protected: string;
  fingerprint: Stage1Fingerprint;
};

export type Stage1Response = {
  results: Stage1PerAttr[];
};

export async function runStage1(
  csv: ParsedCsv,
  cfg: AuditConfig,
  signal?: AbortSignal
): Promise<Stage1Response> {
  const fd = new FormData();
  const blob = new Blob([csv.text], { type: "text/csv" });
  fd.append("file", blob, csv.fileName);
  fd.append("target", cfg.target);
  for (const p of cfg.protectedAttrs) fd.append("protected", p);
  fd.append("positive_class", cfg.positiveClass);

  const res = await fetch(`${API_BASE}/api/audit/stage/1`, {
    method: "POST",
    body: fd,
    signal,
  });

  if (!res.ok) {
    throw new Error(await readError(res, "Stage 1"));
  }
  return res.json();
}

/* ─────────────  Stage 2 (multi-model training via Optuna, streamed)  ───────────── */

export type Stage2ModelStatus = "running" | "done" | "error";

export type Stage2Model = {
  key: "logistic" | "rf" | "xgb" | "lgbm" | "svm";
  name: string;
  family: string;
  color: string;
  status: Stage2ModelStatus;
  best_score: number | null;
  best_params: Record<string, number | string>;
  n_trials: number;
  cv_mean: number | null;
  cv_std: number | null;
  train_time_sec: number | null;
  available: boolean;
  fallback_note: string | null;
  error: string | null;
};

export type Stage2InitEvent = {
  event: "init";
  session_id: string;
  models: Stage2Model[];
  n_train: number;
  n_features: number;
  n_trials_per_model: number;
  cv_folds: number;
  scoring: string;
  xgboost_available: boolean;
  lightgbm_available: boolean;
};

export type Stage2ModelDoneEvent = {
  event: "model_done";
  model: Stage2Model;
};

export type Stage2DoneEvent = {
  event: "done";
  total_train_time_sec: number | null;
};

export type Stage2Event =
  | Stage2InitEvent
  | Stage2ModelDoneEvent
  | Stage2DoneEvent;

/** Streams Stage 2 progress as ndjson — `onEvent` is called for each event
 * (init, one per model_done, then done). Resolves when the stream ends. */
export async function runStage2Stream(
  csv: ParsedCsv,
  cfg: AuditConfig,
  onEvent: (event: Stage2Event) => void,
  signal?: AbortSignal
): Promise<void> {
  const fd = new FormData();
  const blob = new Blob([csv.text], { type: "text/csv" });
  fd.append("file", blob, csv.fileName);
  fd.append("target", cfg.target);
  for (const p of cfg.protectedAttrs) fd.append("protected", p);
  fd.append("positive_class", cfg.positiveClass);

  const res = await fetch(`${API_BASE}/api/audit/stage/2`, {
    method: "POST",
    body: fd,
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(await readError(res, "Stage 2"));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function flushLine(line: string) {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as Stage2Event);
    } catch {
      // Malformed line — skip silently rather than aborting the whole audit.
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      flushLine(buffer.slice(0, nl));
      buffer = buffer.slice(nl + 1);
    }
  }
  if (buffer.length > 0) flushLine(buffer);
}

/* ─────────────  Stage 3 (per-model fairness audit)  ───────────── */

export type Stage3GroupMetrics = {
  n: number;
  tpr: number | null;
  tpr_ci: [number | null, number | null];
  fpr: number | null;
  fpr_ci: [number | null, number | null];
  fnr: number | null;
  tnr: number | null;
  ppv: number | null;
  ppv_ci: [number | null, number | null];
  selection_rate: number | null;
  auc: number | null;
  auc_ci: [number | null, number | null];
  sufficient_power: boolean;
};

export type Stage3Gaps = {
  tpr_gap: number | null;
  fpr_gap: number | null;
  eo_gap: number | null;
  dp_gap: number | null;
  di_ratio: number | null;
  ppv_gap: number | null;
};

export type Stage3PairwiseTest = {
  model_a: string;
  model_b: string;
  mcnemar_p: number | null;
  mcnemar_p_adj: number | null;
  delong_p: number | null;
  delong_p_adj: number | null;
  significant_errors: boolean;
  significant_auc: boolean;
};

export type Stage3Model = {
  key: Stage2Model["key"];
  name: string;
  family: string;
  color: string;
  overall_auc: number | null;
  overall_auc_ci: [number | null, number | null];
  ece: number | null;
  by_group: Record<string, Stage3GroupMetrics>;
  gaps: Stage3Gaps;
};

export type Stage3PerAttr = {
  protected: string;
  groups: string[];
  models: Stage3Model[];
};

export type Stage3Response = {
  session_id: string;
  n_total: number;
  bootstrap_n: number;
  pairwise_tests: Stage3PairwiseTest[];
  results: Stage3PerAttr[];
};

export async function runStage3(
  sessionId: string,
  signal?: AbortSignal
): Promise<Stage3Response> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  const res = await fetch(`${API_BASE}/api/audit/stage/3`, {
    method: "POST",
    body: fd,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res, "Stage 3"));
  return res.json();
}

/* Stage 3 streamed events (NDJSON). Mirrors Stage 2's event protocol so the UI
 * can render models progressively as their fairness audits complete. */

export type Stage3InitEvent = {
  event: "init";
  session_id: string;
  n_total: number;
  bootstrap_n: number;
  /** Skeleton — one per protected attribute, with placeholder "running" models. */
  results: {
    protected: string;
    groups: string[];
    models: {
      key: string;
      name: string;
      family: string;
      color: string;
      status: "running";
    }[];
  }[];
};

export type Stage3ModelDoneEvent = {
  event: "model_done";
  protected: string;
  model_key: string;
  model: Stage3Model;
  error?: string;
};

export type Stage3ModelErrorEvent = {
  event: "model_done";
  model_key: string;
  error: string;
};

export type Stage3PairwiseDoneEvent = {
  event: "pairwise_done";
  tests: Stage3PairwiseTest[];
};

export type Stage3DoneEvent = { event: "done" };

export type Stage3StreamEvent =
  | Stage3InitEvent
  | Stage3ModelDoneEvent
  | Stage3PairwiseDoneEvent
  | Stage3DoneEvent;

export async function runStage3Stream(
  sessionId: string,
  onEvent: (ev: Stage3StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const fd = new FormData();
  fd.append("session_id", sessionId);

  const res = await fetch(`${API_BASE}/api/audit/stage/3/stream`, {
    method: "POST",
    body: fd,
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(await readError(res, "Stage 3"));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function flushLine(line: string) {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as Stage3StreamEvent);
    } catch {
      // Skip malformed lines silently rather than aborting the audit.
    }
  }

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) !== -1) {
      flushLine(buffer.slice(0, nl));
      buffer = buffer.slice(nl + 1);
    }
  }
  if (buffer.length > 0) flushLine(buffer);
}

/* ─────────────  Stage 4 (Pareto frontier)  ───────────── */

export type Stage4Model = {
  key: Stage2Model["key"];
  name: string;
  family: string;
  color: string;
  auc: number | null;
  fairness_gap: number | null;
  tpr_gap: number | null;
  fpr_gap: number | null;
  dp_gap: number | null;
  di_ratio: number | null;
  ppv_gap: number | null;
  pareto_optimal: boolean;
  /** True if EO gap ≤ eo_gap_threshold. Required (in addition to Pareto-optimality)
   *  for a model to be eligible for the recommendation. */
  fairness_qualified: boolean;
  /** AUC − λ·EO_gap. Higher = better tradeoff. Used as an optional ranking signal
   *  inside the Pareto + fairness-qualified set. */
  composite_score: number | null;
  recommended: boolean;
};

export type Stage4PerAttr = {
  protected: string;
  models: Stage4Model[];
  /** Plain-English explanation of why the recommended model was chosen,
   *  mentioning the AUC, EO gap, and threshold. Null if no model is recommended. */
  recommended_reason: string | null;
  /** Surfaced when no model satisfies the fairness threshold; tells the user
   *  what to inspect manually. Null when a recommendation was made. */
  recommendation_warning: string | null;
};

export type Stage4Response = {
  session_id: string;
  /** Equalized-odds gap above which a model is disqualified from being recommended. */
  eo_gap_threshold: number;
  /** Lambda used in the composite score Score = AUC − λ·EO_gap. */
  lambda_param: number;
  results: Stage4PerAttr[];
};

export async function runStage4(
  sessionId: string,
  signal?: AbortSignal
): Promise<Stage4Response> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  const res = await fetch(`${API_BASE}/api/audit/stage/4`, {
    method: "POST",
    body: fd,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res, "Stage 4"));
  return res.json();
}

/* ─────────────  Stage 5 (root cause diagnosis)  ───────────── */

export type Stage5GroupShap = {
  importance: Record<string, number | null>;
  ranks: Record<string, number>;
};

export type Stage5ProxyFeature = {
  feature: string;
  ratio: number | null;
  max_group: string;
  min_group: string;
  rank_delta: number | null;
  importances: Record<string, number | null>;
};

export type Stage5CorrFeature = {
  feature: string;
  smd: number | null;
  mean_a: number | null;
  mean_b: number | null;
  group_a: string;
  group_b: string;
};

export type Stage5PerAttr = {
  groups: string[];
  /** Sample count per raw group value (e.g., {"0": 312, "1": 45}). */
  group_sizes: Record<string, number>;
  /** Positive-rate (fraction with target=positive) per raw group value. */
  group_positive_rates: Record<string, number | null>;
  group_shap: Record<string, Stage5GroupShap>;
  proxy_features: Stage5ProxyFeature[];
  correlated_features: Stage5CorrFeature[];
  permutation_test: {
    p_value: number | null;
    n_permutations: number;
    significant: boolean;
  };
  counterfactual_flip_rate: number | null;
  eo_gap: number | null;
  dp_gap: number | null;
  base_rate_gap: number | null;
  bayesian_root_cause: Record<string, number | null>;
};

export type Stage5Response = {
  session_id: string;
  model_key: string;
  model_name: string;
  shap_available: boolean;
  primary_root_cause: string | null;
  results: Record<string, Stage5PerAttr>;
};

export async function runStage5(
  sessionId: string,
  modelKey?: string,
  signal?: AbortSignal
): Promise<Stage5Response> {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  if (modelKey) fd.append("model_key", modelKey);
  const res = await fetch(`${API_BASE}/api/audit/stage/5`, {
    method: "POST",
    body: fd,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res, "Stage 5"));
  return res.json();
}

/* ─────────────  Stage 6 (guided remediation)  ───────────── */

export type Stage6ActionStatus = "recommended" | "optional" | "blocked";

export type Stage6Action = {
  id: string;
  title: string;
  status: Stage6ActionStatus;
  body: string;
  /** Measurable, bootstrap-validated acceptance criteria for the fix. Empty
   *  for blocked actions (no fix is being recommended). */
  success_criteria: string[];
};

export type Stage6Response = {
  session_id: string;
  model_key: string | null;
  model_name: string;
  primary_root_cause: string;
  diagnosis: string;
  summary: string;
  actions: Stage6Action[];
  safe_to_auto_fix: boolean;
  warning: string | null;
};

export async function runStage6(
  sessionId: string,
  stage5: Stage5Response,
  signal?: AbortSignal
): Promise<Stage6Response> {
  // Lazy-import the encoding decoder so this module stays free of UI deps.
  const { formatGroup, attrLabel } = await import("./labels");

  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("primary_root_cause", stage5.primary_root_cause ?? "");
  fd.append("model_key", stage5.model_key ?? "");
  fd.append("model_name", stage5.model_name ?? "");

  const firstAttrKey = stage5.results ? Object.keys(stage5.results)[0] : null;
  const firstAttr = firstAttrKey ? stage5.results[firstAttrKey] : null;

  if (firstAttr) {
    const confidence =
      firstAttr.bayesian_root_cause[stage5.primary_root_cause ?? ""] ?? 0;
    fd.append("confidence", String(confidence));
    fd.append("eo_gap", String(firstAttr.eo_gap ?? 0));
    fd.append("dp_gap", String(firstAttr.dp_gap ?? 0));
    fd.append("flip_rate", String(firstAttr.counterfactual_flip_rate ?? 0));
    fd.append("proxy_count", String(firstAttr.proxy_features.length));
    const topProxy = firstAttr.proxy_features[0]?.feature;
    if (topProxy) fd.append("top_proxy_feature", topProxy);

    // Identify minority/majority groups via Stage 5's group_sizes so the
    // backend can name them in its remediation templates.
    if (firstAttrKey && firstAttr.group_sizes) {
      const sized = Object.entries(firstAttr.group_sizes)
        .map(([g, n]) => [g, n] as const);
      if (sized.length >= 2) {
        sized.sort((a, b) => a[1] - b[1]);
        const [minRaw, minN] = sized[0];
        const [majRaw, majN] = sized[sized.length - 1];
        const minFmt = formatGroup(firstAttrKey, minRaw);
        const majFmt = formatGroup(firstAttrKey, majRaw);
        fd.append("minority_label", minFmt.full);
        fd.append("majority_label", majFmt.full);
        fd.append("minority_n", String(minN));
        fd.append("majority_n", String(majN));
        fd.append("attr_human_label", attrLabel(firstAttrKey));
      }
    }
  }
  if (firstAttrKey) fd.append("protected_attr", firstAttrKey);

  const res = await fetch(`${API_BASE}/api/audit/stage/6`, {
    method: "POST",
    body: fd,
    signal,
  });
  if (!res.ok) throw new Error(await readError(res, "Stage 6"));
  return res.json();
}

/* ─────────────  Internals  ───────────── */

async function readError(res: Response, label: string): Promise<string> {
  let detail = "";
  try {
    const body = await res.json();
    detail = body?.error ?? "";
  } catch {
    detail = await res.text().catch(() => "");
  }
  const status = `${res.status}${res.statusText ? " " + res.statusText : ""}`;
  return `${label} failed (${status})${detail ? ": " + detail : ""}`;
}
