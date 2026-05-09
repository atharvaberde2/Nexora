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
};

export type Stage3Model = {
  key: Stage2Model["key"];
  name: string;
  family: string;
  color: string;
  overall_auc: number | null;
  overall_auc_ci: [number | null, number | null];
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
  pareto_optimal: boolean;
  recommended: boolean;
};

export type Stage4PerAttr = {
  protected: string;
  models: Stage4Model[];
};

export type Stage4Response = {
  session_id: string;
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
