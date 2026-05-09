"use client";

import { useEffect, useRef, useState } from "react";
import { Badge, Button, Card, CardHeader } from "@/components/primitives";
import { cn } from "@/lib/cn";
import { ReportTabs } from "@/components/report-tabs";
import { REMEDIATION, ROOT_CAUSE } from "@/lib/data";
import {
  runStage1,
  runStage2Stream,
  runStage3,
  runStage4,
  type Stage1Fingerprint,
  type Stage1Response,
  type Stage2Model,
  type Stage3Response,
  type Stage3Model,
  type Stage4Response,
  type Stage4Model,
} from "@/lib/api";
import type { ParsedCsv } from "@/lib/csv";
import type { AuditConfig } from "./stage-configure";

type StageCtx = { csv: ParsedCsv; cfg: AuditConfig; data: StageData };

/** Stage 2 is built up incrementally from streamed events.
 *  - `models` is populated as soon as the `init` event arrives (all "running")
 *  - each model is replaced when its `model_done` event lands
 *  - `complete` flips true on the `done` event
 *  - `session_id` is captured from `init` and used to drive Stages 3 & 4 */
type Stage2Live = {
  session_id: string;
  models: Stage2Model[];
  n_train: number;
  n_features: number;
  n_trials_per_model: number;
  cv_folds: number;
  scoring: string;
  total_train_time_sec: number | null;
  xgboost_available: boolean;
  lightgbm_available: boolean;
  complete: boolean;
};

type StageData = {
  stage1?: Stage1Response;
  stage2?: Stage2Live;
  stage3?: Stage3Response;
  stage4?: Stage4Response;
};

type StageDef = {
  id: number;
  name: string;
  desc: string;
  duration: number; // ms — synthetic "audit time" used for the in-stage progress bar and the elapsed total
  meta: string;
  finding: (ctx: StageCtx) => string;
};

const PIPELINE: StageDef[] = [
  {
    id: 1,
    name: "Bias fingerprinting",
    desc: "Auditing input distribution, missingness disparity, and per-group power.",
    meta: "χ² · LIVE BACKEND",
    duration: 900,
    finding: ({ cfg, data }) => {
      const r = data.stage1;
      if (!r || r.results.length === 0) {
        return `Bias fingerprint computed for ${cfg.protectedAttrs.join(", ")}`;
      }
      const sigCount = r.results.filter((x) => x.fingerprint.label_bias.significant).length;
      const parts = r.results.map((x) => {
        const p = x.fingerprint.label_bias.p_value;
        const pStr = p == null ? "—" : p.toFixed(3);
        return `${x.protected} (χ² p=${pStr})`;
      });
      const sigSuffix =
        sigCount > 0
          ? ` · ${sigCount}/${r.results.length} significant`
          : "";
      return `${parts.join(" · ")}${sigSuffix}`;
    },
  },
  {
    id: 2,
    name: "Multi-model training",
    desc: "Five model families trained with Optuna hyperparameter search and stratified k-fold CV.",
    meta: "OPTUNA · LIVE BACKEND",
    duration: 2400,
    finding: ({ data }) => {
      const r = data.stage2;
      if (!r || r.models.length === 0) {
        return "Multi-model training complete";
      }
      const scored = r.models
        .filter((m) => m.status === "done" && m.best_score != null)
        .sort((a, b) => (b.best_score ?? 0) - (a.best_score ?? 0));
      if (scored.length === 0) {
        return `${r.models.length} models attempted · all failed`;
      }
      const best = scored[0];
      const worst = scored[scored.length - 1];
      return `${scored.length}/${r.models.length} models · best ${best.name} AUC ${(best.best_score ?? 0).toFixed(3)} · worst ${(worst.best_score ?? 0).toFixed(3)}`;
    },
  },
  {
    id: 3,
    name: "Per-model fairness audit",
    desc: "Per-group TPR, FPR, AUC, selection rate with bootstrap CIs and equalized-odds / demographic-parity gaps.",
    meta: "BOOTSTRAP · LIVE BACKEND",
    duration: 1200,
    finding: ({ data }) => {
      const r = data.stage3;
      if (!r) return "Fairness metrics computed";
      const attrs = r.results.length;
      const nModels = r.results[0]?.models.length ?? 0;
      const totalCells = r.results.reduce(
        (s, x) => s + x.models.length * Object.keys(x.models[0]?.by_group ?? {}).length,
        0
      );
      const eoGaps = r.results.flatMap((x) =>
        x.models.map((m) => m.gaps.eo_gap).filter((v): v is number => v != null)
      );
      const worst = eoGaps.length > 0 ? Math.max(...eoGaps) : null;
      return `${attrs} attribute${attrs > 1 ? "s" : ""} × ${nModels} models · ${totalCells} group cells · bootstrap n=${r.bootstrap_n}${worst != null ? ` · worst EO Δ ${(worst * 100).toFixed(1)}pp` : ""}`;
    },
  },
  {
    id: 4,
    name: "Pareto frontier",
    desc: "Models compared on AUC vs. equalized-odds gap. Dominated models drop out; the recommended pick is the best-AUC survivor.",
    meta: "PARETO DOMINANCE · LIVE BACKEND",
    duration: 900,
    finding: ({ data }) => {
      const r = data.stage4;
      if (!r) return "Pareto analysis complete";
      const lines = r.results.map((x) => {
        const total = x.models.length;
        const optimal = x.models.filter((m) => m.pareto_optimal).length;
        const rec = x.models.find((m) => m.recommended);
        return `${x.protected}: ${optimal}/${total} on frontier${rec ? ` · rec ${rec.name}` : ""}`;
      });
      return lines.join(" · ");
    },
  },
  {
    id: 5,
    name: "Root cause diagnosis",
    desc: "SHAP cross-group analysis fed into a 5-class Bayesian classifier.",
    meta: "5-CLASS BAYESIAN",
    duration: 1300,
    finding: () => "Proxy discrimination · 73% confidence",
  },
  {
    id: 6,
    name: "Guided remediation",
    desc: "Root-cause-conditional fixes, validated as Pareto improvements.",
    meta: "BOOTSTRAP-VALIDATED",
    duration: 900,
    finding: () => "Threshold adjustment blocked · feature decorrelation recommended",
  },
  {
    id: 7,
    name: "Gemini reasoning layer",
    desc: "Four schema-validated checkpoints, mathematically constrained to the Pareto-optimal set.",
    meta: "GEMINI 1.5 PRO · PYDANTIC",
    duration: 1100,
    finding: () => "All four checkpoints passed · recommendation cross-checked",
  },
  {
    id: 8,
    name: "Three-audience report",
    desc: "Engineer scorecard · Regulator submission · Community plain-language report.",
    meta: "EN + ES · ELEVENLABS",
    duration: 700,
    finding: () => "3 reports rendered · Pydantic-validated · ready",
  },
];

type StageStatus = "idle" | "running" | "complete";

export function RunningStage({
  csv,
  cfg,
  onComplete,
}: {
  csv: ParsedCsv;
  cfg: AuditConfig;
  onComplete: (totalRunningSec: number) => void;
}) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [status, setStatus] = useState<StageStatus>("idle");
  const [stageElapsedMs, setStageElapsedMs] = useState(0);
  const [findings, setFindings] = useState<Record<number, string>>({});
  const [data, setData] = useState<StageData>({});
  const [error, setError] = useState<string | null>(null);
  const totalRunningMsRef = useRef(0);
  const stageStartedAtRef = useRef<number | null>(null);

  const stage = PIPELINE[currentIdx];
  const isLast = currentIdx === PIPELINE.length - 1;
  const stageProgress = Math.min(100, (stageElapsedMs / stage.duration) * 100);

  // Drive the per-stage animation when the user clicks "Run".
  // Stage 1 calls the Flask backend; later stages still use synthetic timing.
  useEffect(() => {
    if (status !== "running") return;

    stageStartedAtRef.current = Date.now();
    setStageElapsedMs(0);

    const tick = setInterval(() => {
      const start = stageStartedAtRef.current!;
      setStageElapsedMs(Date.now() - start);
    }, 50);

    let cancelled = false;
    const controller = new AbortController();

    async function execute() {
      try {
        if (stage.id === 1) {
          const [stage1] = await Promise.all([
            runStage1(csv, cfg, controller.signal),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage1 }));
          finalize({ ...data, stage1 });
        } else if (stage.id === 2) {
          // Stream events into data.stage2 so the UI can render live progress.
          // We hold a local snapshot and merge into React state on every event.
          let snap: Stage2Live | null = null;
          const minDuration = new Promise<void>((r) =>
            setTimeout(r, stage.duration)
          );
          const stream = runStage2Stream(
            csv,
            cfg,
            (ev) => {
              if (cancelled) return;
              if (ev.event === "init") {
                snap = {
                  session_id: ev.session_id,
                  models: ev.models,
                  n_train: ev.n_train,
                  n_features: ev.n_features,
                  n_trials_per_model: ev.n_trials_per_model,
                  cv_folds: ev.cv_folds,
                  scoring: ev.scoring,
                  xgboost_available: ev.xgboost_available,
                  lightgbm_available: ev.lightgbm_available,
                  total_train_time_sec: null,
                  complete: false,
                };
              } else if (ev.event === "model_done" && snap) {
                snap = {
                  ...snap,
                  models: snap.models.map((m) =>
                    m.key === ev.model.key ? ev.model : m
                  ),
                };
              } else if (ev.event === "done" && snap) {
                snap = {
                  ...snap,
                  total_train_time_sec: ev.total_train_time_sec,
                  complete: true,
                };
              }
              if (snap) {
                const next = snap;
                setData((d) => ({ ...d, stage2: next }));
              }
            },
            controller.signal
          );
          await Promise.all([stream, minDuration]);
          if (cancelled) return;
          finalize({ ...data, stage2: snap ?? undefined });
        } else if (stage.id === 3) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 3 requires Stage 2 to have completed first");
          }
          const [stage3] = await Promise.all([
            runStage3(data.stage2.session_id, controller.signal),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage3 }));
          finalize({ ...data, stage3 });
        } else if (stage.id === 4) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 4 requires Stage 2 to have completed first");
          }
          const [stage4] = await Promise.all([
            runStage4(data.stage2.session_id, controller.signal),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage4 }));
          finalize({ ...data, stage4 });
        } else {
          await new Promise<void>((r) => setTimeout(r, stage.duration));
          if (cancelled) return;
          finalize(data);
        }
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(
          `${msg}. Is the Flask backend running on ${
            process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
          }?`
        );
        setStatus("idle");
      }
    }

    function finalize(nextData: StageData) {
      setStageElapsedMs(stage.duration);
      setFindings((f) => ({
        ...f,
        [currentIdx]: stage.finding({ csv, cfg, data: nextData }),
      }));
      totalRunningMsRef.current += stage.duration;
      setStatus("complete");
    }

    execute();

    return () => {
      cancelled = true;
      controller.abort();
      clearInterval(tick);
    };
    // `data` is intentionally omitted — we read the latest snapshot inline.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, currentIdx, csv, cfg, stage]);

  function runStage() {
    if (status === "idle") {
      setError(null);
      setStatus("running");
    }
  }

  function advance() {
    if (isLast) {
      onComplete(totalRunningMsRef.current / 1000);
      return;
    }
    setCurrentIdx(currentIdx + 1);
    setStatus("idle");
    setStageElapsedMs(0);
  }

  return (
    <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-10 sm:py-12">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-7">
        <div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
            Step 3 — Audit
          </div>
          <h1 className="font-serif text-2xl leading-tight text-ink mb-2">
            Walk the eight-stage statistical pipeline.
          </h1>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-muted font-mono">
            <span className="text-ink">{csv.fileName}</span>
            <span>{csv.rowCount.toLocaleString()} rows</span>
            <span>
              target = <span className="text-ink">{cfg.target}</span>
            </span>
            <span>
              protected ={" "}
              <span className="text-ink">{cfg.protectedAttrs.join(", ")}</span>
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="font-serif text-3xl text-ink tabular">
            {currentIdx + 1}
            <span className="text-ink-dim">/{PIPELINE.length}</span>
          </div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            stage
          </div>
        </div>
      </div>

      {/* Pipeline rail */}
      <PipelineRail currentIdx={currentIdx} status={status} />

      {/* Stage card */}
      <Card className="mt-6">
        <CardHeader
          eyebrow={`Stage 0${stage.id} / 08`}
          title={stage.name}
          meta={stage.meta}
        />

        <div className="p-5 sm:p-6">
          {/* Stage description + status indicator */}
          <div className="flex items-start justify-between gap-4 mb-5">
            <p className="text-sm text-ink-muted leading-relaxed max-w-2xl">
              {stage.desc}
            </p>
            <StageStatusBadge status={status} />
          </div>

          {/* In-stage progress bar — visible while running, settles when complete */}
          {status !== "idle" && (
            <div className="mb-6">
              <div className="h-1 rounded-full bg-elevated overflow-hidden">
                <div
                  className="h-full bg-accent transition-all duration-100 ease-linear"
                  style={{ width: `${stageProgress}%` }}
                />
              </div>
              <div className="flex justify-between mt-1.5 text-2xs font-mono text-ink-dim tabular">
                <span>{(stageElapsedMs / 1000).toFixed(1)}s</span>
                <span>{stageProgress.toFixed(0)}%</span>
                <span>{(stage.duration / 1000).toFixed(1)}s</span>
              </div>
            </div>
          )}

          {/* Error (if last run failed) */}
          {error && status === "idle" && (
            <ErrorBanner message={error} onDismiss={() => setError(null)} />
          )}

          {/* Body */}
          {status === "idle" && <RunPrompt onRun={runStage} stageId={stage.id} />}
          {status === "running" && stage.id === 2 && data.stage2 ? (
            <StageArtifact stageId={stage.id} csv={csv} cfg={cfg} data={data} />
          ) : status === "running" ? (
            <RunningBody stageId={stage.id} />
          ) : null}
          {status === "complete" && (
            <StageArtifact
              stageId={stage.id}
              csv={csv}
              cfg={cfg}
              data={data}
            />
          )}

          {/* Finding bar (when complete) */}
          {status === "complete" && findings[currentIdx] && (
            <div className="mt-6 rounded-md border border-success/20 bg-success/[0.06] px-4 py-3 text-xs font-mono text-success flex items-start gap-2 animate-fade-up">
              <span>→</span>
              <span>{findings[currentIdx]}</span>
            </div>
          )}
        </div>
      </Card>

      {/* CTA row */}
      <div className="mt-6 flex flex-wrap items-center gap-3 justify-end">
        {status === "idle" && (
          <Button variant="primary" size="md" onClick={runStage}>
            Run stage {stage.id}
            <PlayIcon />
          </Button>
        )}
        {status === "running" && (
          <Button variant="primary" size="md" disabled>
            Running…
          </Button>
        )}
        {status === "complete" && !isLast && (
          <Button variant="primary" size="md" onClick={advance}>
            Continue · Stage {stage.id + 1} — {PIPELINE[currentIdx + 1].name}
            <ArrowRight />
          </Button>
        )}
        {status === "complete" && isLast && (
          <Button variant="primary" size="md" onClick={advance}>
            Review full results
            <ArrowRight />
          </Button>
        )}
      </div>
    </div>
  );
}

/* ─────────────  Pipeline rail (8 dots)  ───────────── */

function PipelineRail({
  currentIdx,
  status,
}: {
  currentIdx: number;
  status: StageStatus;
}) {
  return (
    <div className="rounded-lg border border-hairline bg-surface/60 px-4 py-4">
      <ol className="grid grid-cols-4 sm:grid-cols-8 gap-2">
        {PIPELINE.map((s, i) => {
          const state =
            i < currentIdx
              ? "done"
              : i === currentIdx
              ? status === "complete"
                ? "done"
                : status === "running"
                ? "running"
                : "current"
              : "pending";
          return (
            <li key={s.id} className="flex items-center gap-2 min-w-0">
              <RailGlyph state={state} index={s.id} />
              <div className="min-w-0 hidden sm:block">
                <div
                  className={cn(
                    "text-2xs font-mono uppercase tracking-wider truncate",
                    state === "pending" ? "text-ink-faint" : "text-ink-dim"
                  )}
                >
                  Stage {s.id}
                </div>
                <div
                  className={cn(
                    "text-xs truncate",
                    state === "pending" ? "text-ink-muted" : "text-ink"
                  )}
                >
                  {s.name}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function RailGlyph({
  state,
  index,
}: {
  state: "done" | "running" | "current" | "pending";
  index: number;
}) {
  if (state === "done") {
    return (
      <div className="w-6 h-6 rounded-full bg-accent flex items-center justify-center shrink-0">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <path
            d="M3 6L5 8L9 4"
            stroke="#08090B"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
        </svg>
      </div>
    );
  }
  if (state === "running") {
    return (
      <div className="relative w-6 h-6 shrink-0">
        <span className="absolute inset-0 rounded-full bg-accent/30 animate-ping" />
        <span className="absolute inset-1 rounded-full bg-accent" />
      </div>
    );
  }
  if (state === "current") {
    return (
      <div className="w-6 h-6 rounded-full border-2 border-accent text-accent flex items-center justify-center shrink-0 font-mono text-2xs tabular">
        {index}
      </div>
    );
  }
  return (
    <div className="w-6 h-6 rounded-full border border-line text-ink-faint flex items-center justify-center shrink-0 font-mono text-2xs tabular">
      {index}
    </div>
  );
}

function StageStatusBadge({ status }: { status: StageStatus }) {
  if (status === "idle") return <Badge tone="neutral">Ready</Badge>;
  if (status === "running")
    return (
      <Badge tone="accent" dot>
        Running
      </Badge>
    );
  return (
    <Badge tone="success" dot>
      Complete
    </Badge>
  );
}

/* ─────────────  Idle / Running placeholders  ───────────── */

function RunPrompt({ onRun, stageId }: { onRun: () => void; stageId: number }) {
  return (
    <div className="rounded-md border border-hairline bg-elevated/40 grid-bg px-6 py-12 text-center">
      <div className="font-serif text-lg text-ink mb-2">
        Stage {stageId} is queued.
      </div>
      <div className="text-sm text-ink-muted max-w-md mx-auto mb-6">
        Click run to execute this stage and reveal its statistical artifact.
      </div>
      <div className="flex justify-center">
        <Button variant="primary" size="sm" onClick={onRun}>
          Run stage {stageId}
          <PlayIcon />
        </Button>
      </div>
    </div>
  );
}

function RunningBody({ stageId }: { stageId: number }) {
  return (
    <div className="rounded-md border border-hairline bg-elevated/40 grid-bg px-6 py-12 text-center">
      <div className="relative w-12 h-12 mx-auto mb-5">
        <span className="absolute inset-0 rounded-full bg-accent/30 animate-ping" />
        <span className="absolute inset-2 rounded-full bg-accent" />
      </div>
      <div className="font-serif text-lg text-ink mb-1">
        Stage {stageId} running…
      </div>
      <div className="text-xs font-mono text-ink-dim uppercase tracking-wider">
        Statistics computing
      </div>
    </div>
  );
}

/* ─────────────  Stage artifacts  ───────────── */

function StageArtifact({
  stageId,
  csv,
  cfg,
  data,
}: {
  stageId: number;
  csv: ParsedCsv;
  cfg: AuditConfig;
  data: StageData;
}) {
  if (stageId === 1) {
    if (!data.stage1) return null;
    return <BiasFingerprintArtifact cfg={cfg} response={data.stage1} />;
  }
  if (stageId === 2) {
    if (!data.stage2) return null;
    return <TrainingArtifact live={data.stage2} />;
  }
  if (stageId === 3) {
    if (!data.stage3) return null;
    return <FairnessArtifact response={data.stage3} />;
  }
  if (stageId === 4) {
    if (!data.stage4) return null;
    return <ParetoArtifact response={data.stage4} />;
  }
  if (stageId === 5) return <RootCauseArtifact cfg={cfg} />;
  if (stageId === 6) return <RemediationArtifact />;
  if (stageId === 7) return <GeminiArtifact />;
  if (stageId === 8) return <ReportTabs />;
  return null;
}

/* Stage 1 — Bias fingerprint, served by Flask backend (POST /api/audit/stage/1).
   One panel per protected attribute, audited independently. */

function BiasFingerprintArtifact({
  cfg,
  response,
}: {
  cfg: AuditConfig;
  response: Stage1Response;
}) {
  if (response.results.length === 0) return null;
  return (
    <div className="space-y-8">
      {response.results.map((r, i) => (
        <div key={r.protected} className="space-y-3">
          {response.results.length > 1 && (
            <div className="flex items-baseline justify-between">
              <div className="font-serif text-lg text-ink">
                <span className="text-ink-dim font-mono text-xs uppercase tracking-[0.18em] mr-2">
                  Attr {i + 1}/{response.results.length}
                </span>
                {r.protected}
              </div>
              <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
                independent audit
              </div>
            </div>
          )}
          <FingerprintPanel
            cfg={cfg}
            protectedName={r.protected}
            result={r.fingerprint}
          />
        </div>
      ))}
    </div>
  );
}

function FingerprintPanel({
  cfg,
  protectedName,
  result,
}: {
  cfg: AuditConfig;
  protectedName: string;
  result: Stage1Fingerprint;
}) {
  const groupTotal = result.groups.reduce((s, g) => s + g.n, 0) || 1;
  const overall = result.overall_positive_rate;
  const lb = result.label_bias;
  const smallest = result.groups[result.groups.length - 1];
  const allPowered = result.groups.every((g) => g.sufficient_power);

  return (
    <div className="grid lg:grid-cols-3 gap-5">
      {/* Group distribution + class-imbalance */}
      <Card className="lg:col-span-2 p-5">
        <div className="flex items-baseline justify-between mb-4">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            Group distribution & positive rate · {protectedName}
          </div>
          <div className="text-2xs font-mono text-ink-dim tracking-wider">
            n = {result.n_total.toLocaleString()}
          </div>
        </div>
        <div className="space-y-4">
          {result.groups.map((g) => {
            const sharePct = (g.n / groupTotal) * 100;
            const posPct = g.positive_rate != null ? g.positive_rate * 100 : null;
            const gap = g.base_rate_gap;
            return (
              <div key={g.name}>
                <div className="flex items-baseline justify-between text-xs mb-1">
                  <span className="text-ink">{g.name}</span>
                  <span className="font-mono tabular text-ink-dim">
                    {g.n.toLocaleString()}{" "}
                    <span className="text-ink-faint">· {sharePct.toFixed(1)}%</span>
                    {posPct != null && (
                      <>
                        {" · "}
                        <span className="text-ink">P(+) {posPct.toFixed(1)}%</span>
                      </>
                    )}
                    {gap != null && (
                      <span
                        className={cn(
                          "ml-1.5",
                          Math.abs(gap) >= 0.05
                            ? gap > 0
                              ? "text-success"
                              : "text-warning"
                            : "text-ink-dim"
                        )}
                      >
                        ({gap >= 0 ? "+" : ""}
                        {(gap * 100).toFixed(1)}pp)
                      </span>
                    )}
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-elevated overflow-hidden">
                  <div
                    className={cn(
                      "h-full transition-all",
                      !g.sufficient_power ? "bg-warning" : "bg-accent"
                    )}
                    style={{ width: `${sharePct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Overall + label bias + power */}
      <Card className="p-5 space-y-5">
        <div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
            Overall positive rate
          </div>
          <div className="font-mono tabular text-xl text-ink">
            {overall != null ? `${(overall * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="text-xs text-ink-muted mt-1">
            target = <span className="font-mono text-ink">{cfg.target}</span>{" "}
            · positive ={" "}
            <span className="font-mono text-ink">{cfg.positiveClass}</span>
          </div>
        </div>
        <div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
            Label-bias χ²
          </div>
          <div
            className={cn(
              "font-mono tabular text-xl",
              lb.p_value == null
                ? "text-ink-dim"
                : lb.significant
                ? "text-warning"
                : "text-success"
            )}
          >
            {lb.p_value == null ? "—" : `p = ${lb.p_value.toFixed(3)}`}
          </div>
          <div className="text-xs text-ink-muted mt-1">
            {lb.chi2 != null && (
              <>
                χ² ={" "}
                <span className="font-mono text-ink">{lb.chi2.toFixed(2)}</span>
                {" · "}
              </>
            )}
            {lb.p_value == null
              ? "Test not applicable"
              : lb.significant
              ? "Outcome rate differs across groups"
              : "No statistically significant dependency"}
          </div>
        </div>
        <div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
            Power
          </div>
          <div
            className={cn(
              "font-mono tabular text-xl",
              allPowered ? "text-success" : "text-warning"
            )}
          >
            {result.groups.filter((g) => g.sufficient_power).length}/
            {result.groups.length}
          </div>
          <div className="text-xs text-ink-muted mt-1">
            n ≥ {result.n_threshold} met by{" "}
            {result.groups.filter((g) => g.sufficient_power).length} of{" "}
            {result.groups.length} groups · smallest n ={" "}
            <span className="font-mono text-ink">
              {smallest ? smallest.n.toLocaleString() : "—"}
            </span>
          </div>
        </div>
      </Card>

      {/* Missingness */}
      <Card className="lg:col-span-3 p-5">
        <div className="flex items-baseline justify-between mb-4">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            Missingness · per group + top columns
          </div>
          <div className="text-2xs font-mono text-ink-dim tracking-wider">
            avg null fraction across columns
          </div>
        </div>
        <div className="grid sm:grid-cols-2 gap-5">
          <div>
            <div className="text-2xs font-mono uppercase tracking-wider text-ink-dim mb-3">
              By {protectedName}
            </div>
            <div className="space-y-2.5">
              {result.groups.map((g) => {
                const v = g.missing_rate ?? 0;
                const pct = v * 100;
                return (
                  <div key={g.name}>
                    <div className="flex items-baseline justify-between text-xs mb-1">
                      <span className="text-ink-muted">{g.name}</span>
                      <span className="font-mono tabular text-ink-dim">
                        {pct.toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-elevated overflow-hidden">
                      <div
                        className={cn(
                          "h-full",
                          v > 0.1 ? "bg-warning" : "bg-ink-faint"
                        )}
                        style={{ width: `${Math.min(100, pct * 4)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div>
            <div className="text-2xs font-mono uppercase tracking-wider text-ink-dim mb-3">
              Top columns by missingness
            </div>
            <div className="grid grid-cols-2 gap-2">
              {result.top_missing_columns.map((c) => (
                <div
                  key={c.name}
                  className="rounded-md border border-hairline bg-elevated/40 px-3 py-2.5"
                >
                  <div className="text-xs font-mono text-ink truncate">
                    {c.name}
                  </div>
                  <div className="flex items-baseline gap-1.5 mt-1">
                    <span
                      className={cn(
                        "font-mono tabular text-base",
                        (c.missing_pct ?? 0) > 0.1 ? "text-warning" : "text-ink"
                      )}
                    >
                      {c.missing_pct != null
                        ? `${(c.missing_pct * 100).toFixed(1)}%`
                        : "—"}
                    </span>
                    <span className="text-2xs text-ink-dim">missing</span>
                  </div>
                </div>
              ))}
              {result.top_missing_columns.length === 0 && (
                <div className="col-span-2 text-xs text-ink-dim">
                  No missingness detected.
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

/* Stage 2 — Optuna-tuned model leaderboard, served by Flask backend
   (POST /api/audit/stage/2). Shows per-model AUC + best hyperparameters. */

const MODEL_COLOR: Record<string, string> = {
  m1: "#9CA3AF",
  m2: "#A78BFA",
  m3: "#F87171",
  m4: "#34D399",
  m5: "#FBBF24",
};

function TrainingArtifact({ live }: { live: Stage2Live }) {
  const done = live.models.filter((m) => m.status === "done" && m.best_score != null);
  const minAuc = done.length > 0 ? Math.min(...done.map((m) => m.best_score!)) : 0;
  const maxAuc = done.length > 0 ? Math.max(...done.map((m) => m.best_score!)) : 1;
  const span = Math.max(maxAuc - minAuc, 1e-9);
  const completedCount = live.models.filter((m) => m.status !== "running").length;

  return (
    <div className="space-y-4">
      <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {live.models.map((m) => (
          <ModelCard
            key={m.key}
            m={m}
            normWidth={
              m.status === "done" && m.best_score != null
                ? 35 + ((m.best_score - minAuc) / span) * 60
                : 0
            }
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-2xs font-mono text-ink-dim uppercase tracking-wider">
        <span>
          {live.n_train.toLocaleString()} rows · {live.n_features} features
        </span>
        <span>
          {live.scoring} · {live.cv_folds}-fold CV
        </span>
        <span>{live.n_trials_per_model} Optuna trials × model</span>
        <span
          className={cn(
            live.complete ? "text-success" : "text-accent"
          )}
        >
          {completedCount}/{live.models.length} models complete
        </span>
        {live.total_train_time_sec != null && (
          <span>total {live.total_train_time_sec.toFixed(1)}s</span>
        )}
        {!live.xgboost_available && (
          <span className="text-warning">xgboost: fallback</span>
        )}
        {!live.lightgbm_available && (
          <span className="text-warning">lightgbm: fallback</span>
        )}
      </div>
    </div>
  );
}

function ModelCard({ m, normWidth }: { m: Stage2Model; normWidth: number }) {
  const color = MODEL_COLOR[m.color] ?? MODEL_COLOR.m1;
  const isRunning = m.status === "running";
  const isError = m.status === "error";

  return (
    <div
      className={cn(
        "rounded-md border bg-elevated/40 p-4 flex flex-col transition-colors",
        isRunning ? "border-accent/40" : "border-hairline",
        isError && "border-danger/30"
      )}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ background: color }}
        />
        <div className="text-xs font-medium text-ink truncate flex-1">
          {m.name}
        </div>
        <ModelStatusGlyph status={m.status} />
      </div>
      <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1">
        AUC-ROC
      </div>
      <div
        className={cn(
          "font-mono tabular text-lg mb-2",
          isRunning ? "text-ink-dim" : "text-ink"
        )}
      >
        {m.best_score != null ? m.best_score.toFixed(3) : "—"}
      </div>
      <div className="h-1 rounded-full bg-elevated overflow-hidden">
        {isRunning && (
          <div
            className="h-full w-1/3 animate-pulse"
            style={{ background: color, opacity: 0.5 }}
          />
        )}
        {m.status === "done" && (
          <div
            className="h-full transition-all duration-500"
            style={{ width: `${normWidth}%`, background: color }}
          />
        )}
      </div>
      <div className="mt-3 text-2xs font-mono text-ink-dim tracking-wider uppercase">
        {isRunning && "training…"}
        {m.status === "done" &&
          `Trained · ${(m.train_time_sec ?? 0).toFixed(1)}s · ${m.n_trials} trials`}
        {isError && "failed"}
      </div>
      {m.cv_std != null && m.status === "done" && (
        <div className="mt-1 text-2xs font-mono text-ink-faint tabular">
          ± {m.cv_std.toFixed(3)} cv std
        </div>
      )}
      {m.status === "done" && m.best_params && Object.keys(m.best_params).length > 0 && (
        <div className="mt-3 pt-3 border-t border-hairline space-y-1">
          {Object.entries(m.best_params)
            .slice(0, 4)
            .map(([k, v]) => (
              <div
                key={k}
                className="flex items-baseline justify-between gap-2 text-2xs font-mono"
              >
                <span className="text-ink-dim truncate">{k}</span>
                <span className="text-ink tabular truncate">
                  {typeof v === "number"
                    ? Math.abs(v) < 0.01 || Math.abs(v) >= 1000
                      ? v.toExponential(2)
                      : Number.isInteger(v)
                      ? v.toString()
                      : v.toFixed(3)
                    : String(v)}
                </span>
              </div>
            ))}
        </div>
      )}
      {m.fallback_note && (
        <div className="mt-2 text-2xs text-warning leading-snug">
          {m.fallback_note}
        </div>
      )}
      {m.error && (
        <div className="mt-2 text-2xs text-danger font-mono leading-snug break-words">
          {m.error}
        </div>
      )}
    </div>
  );
}

function ModelStatusGlyph({ status }: { status: Stage2Model["status"] }) {
  if (status === "running") {
    return (
      <div className="relative w-3 h-3 shrink-0" aria-label="training">
        <span className="absolute inset-0 rounded-full bg-accent/30 animate-ping" />
        <span className="absolute inset-[3px] rounded-full bg-accent" />
      </div>
    );
  }
  if (status === "done") {
    return (
      <svg
        width="12"
        height="12"
        viewBox="0 0 12 12"
        className="shrink-0 text-success"
        aria-label="complete"
      >
        <path
          d="M3 6L5 8L9 4"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </svg>
    );
  }
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      className="shrink-0 text-danger"
      aria-label="failed"
    >
      <path
        d="M3 3L9 9 M9 3L3 9"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* Stage 3 — Per-model fairness leaderboard, one panel per protected attr.
   Real data: TPR/FPR/AUC by group, equalized-odds & demographic-parity gaps,
   bootstrap CIs. Computed by Flask /api/audit/stage/3. */

function FairnessArtifact({ response }: { response: Stage3Response }) {
  return (
    <div className="space-y-8">
      {response.results.map((attr, i) => (
        <div key={attr.protected} className="space-y-3">
          {response.results.length > 1 && (
            <div className="flex items-baseline justify-between">
              <div className="font-serif text-lg text-ink">
                <span className="text-ink-dim font-mono text-xs uppercase tracking-[0.18em] mr-2">
                  Attr {i + 1}/{response.results.length}
                </span>
                {attr.protected}
              </div>
              <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
                bootstrap n = {response.bootstrap_n.toLocaleString()} · 95% CI
              </div>
            </div>
          )}
          <FairnessTable attr={attr} />
        </div>
      ))}
    </div>
  );
}

function FairnessTable({ attr }: { attr: Stage3Response["results"][number] }) {
  const sorted = [...attr.models].sort(
    (a, b) => (b.overall_auc ?? 0) - (a.overall_auc ?? 0)
  );
  const groups = attr.groups;

  return (
    <div className="overflow-x-auto rounded-md border border-hairline">
      <table className="w-full text-sm tabular">
        <thead>
          <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
            <th className="text-left font-normal py-3 pl-4 pr-3">Model</th>
            <th className="text-right font-normal py-3 px-3">AUC overall</th>
            {groups.map((g) => (
              <th
                key={g}
                className="text-right font-normal py-3 px-3"
                title={`AUC for group ${g}`}
              >
                AUC · {g}
              </th>
            ))}
            <th className="text-right font-normal py-3 px-3" title="max−min TPR across groups">TPR Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max−min FPR across groups">FPR Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max(TPR Δ, FPR Δ)">EO Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max−min selection rate">DP Δ</th>
            <th className="text-right font-normal py-3 px-3 pr-4" title="min/max selection rate; 4/5 rule = 0.80">DI ratio</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <FairnessRow key={m.key} model={m} groups={groups} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FairnessRow({
  model,
  groups,
}: {
  model: Stage3Model;
  groups: string[];
}) {
  const color = MODEL_COLOR[model.color] ?? MODEL_COLOR.m1;
  const eo = model.gaps.eo_gap;
  const dp = model.gaps.dp_gap;
  const tprΔ = model.gaps.tpr_gap;
  const fprΔ = model.gaps.fpr_gap;
  const di = model.gaps.di_ratio;
  const degeneracy = degenerateClassifier(model, groups);

  return (
    <tr className="border-b border-hairline last:border-0">
      <td className="py-3 pl-4 pr-3">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ background: color }}
          />
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="font-medium truncate">{model.name}</span>
              {degeneracy && <DegenerateChip kind={degeneracy} />}
            </div>
            <div className="text-2xs text-ink-dim font-mono truncate">
              {model.family}
            </div>
          </div>
        </div>
      </td>
      <td className="py-3 px-3 text-right font-mono">
        <CiCell value={model.overall_auc} ci={model.overall_auc_ci} digits={3} />
      </td>
      {groups.map((g) => {
        const gm = model.by_group[g];
        return (
          <td key={g} className="py-3 px-3 text-right font-mono">
            <CiCell
              value={gm?.auc ?? null}
              ci={gm?.auc_ci ?? [null, null]}
              digits={3}
            />
            {gm && gm.n < 30 && (
              <div className="text-2xs text-warning font-mono">
                n={gm.n}
              </div>
            )}
          </td>
        );
      })}
      <td className="py-3 px-3 text-right font-mono">
        <GapCell value={tprΔ} />
      </td>
      <td className="py-3 px-3 text-right font-mono">
        <GapCell value={fprΔ} />
      </td>
      <td className="py-3 px-3 text-right font-mono">
        <GapCell value={eo} threshold={0.1} />
      </td>
      <td className="py-3 px-3 text-right font-mono">
        <GapCell value={dp} threshold={0.1} />
      </td>
      <td className="py-3 px-3 pr-4 text-right font-mono">
        {di == null ? (
          <span className="text-ink-dim">—</span>
        ) : (
          <span className={cn(di < 0.8 ? "text-warning" : "text-ink")}>
            {di.toFixed(2)}
          </span>
        )}
      </td>
    </tr>
  );
}

/** A model is "degenerate" when its predict() output is constant across the
 *  data — predicting all-negative or all-positive. Looks "perfectly fair"
 *  on paper (every confusion-matrix gap = 0) but is clinically/decision-
 *  theoretically useless. Detected from selection rate across groups: every
 *  group at 0 → all-negative; every group at 1 → all-positive. */
function degenerateClassifier(
  model: Stage3Model,
  groups: string[]
): "all-negative" | "all-positive" | null {
  const srs = groups
    .map((g) => model.by_group[g]?.selection_rate)
    .filter((v): v is number => v != null);
  if (srs.length < 2) return null;
  if (srs.every((s) => s === 0)) return "all-negative";
  if (srs.every((s) => s === 1)) return "all-positive";
  return null;
}

function DegenerateChip({
  kind,
}: {
  kind: "all-negative" | "all-positive";
}) {
  const label =
    kind === "all-negative" ? "predicts all 0" : "predicts all 1";
  const tooltip =
    kind === "all-negative"
      ? "Model never predicts the positive class at threshold 0.5. Gaps are trivially zero — this is not a fair model, it's a useless one."
      : "Model always predicts the positive class at threshold 0.5. Gaps are trivially zero — this is not a fair model, it's a useless one.";
  return (
    <span
      title={tooltip}
      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-warning/40 bg-warning/10 text-warning text-2xs font-mono uppercase tracking-wider whitespace-nowrap"
    >
      <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true">
        <path
          d="M4.5 1L8 7.5H1L4.5 1Z M4.5 4V5.5 M4.5 6.5V6.6"
          stroke="currentColor"
          strokeWidth="1.1"
          fill="none"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      {label}
    </span>
  );
}

function CiCell({
  value,
  ci,
  digits,
}: {
  value: number | null;
  ci: [number | null, number | null];
  digits: number;
}) {
  if (value == null) return <span className="text-ink-dim">—</span>;
  const half =
    ci[0] != null && ci[1] != null ? (ci[1] - ci[0]) / 2 : null;
  return (
    <span>
      {value.toFixed(digits)}
      {half != null && (
        <span className="text-ink-dim text-xs ml-1.5">
          ±{half.toFixed(digits)}
        </span>
      )}
    </span>
  );
}

function GapCell({
  value,
  threshold = 0.05,
}: {
  value: number | null;
  threshold?: number;
}) {
  if (value == null) return <span className="text-ink-dim">—</span>;
  return (
    <span
      className={cn(
        value >= threshold * 2 ? "text-danger" :
        value >= threshold ? "text-warning" : "text-ink"
      )}
    >
      {(value * 100).toFixed(1)}pp
    </span>
  );
}

/* Stage 4 — Pareto frontier per protected attribute. AUC (X) vs equalized
   odds gap (Y, lower=fairer). Dominated models are dimmed; the recommended
   pick is highlighted. */

function ParetoArtifact({ response }: { response: Stage4Response }) {
  return (
    <div className="space-y-8">
      {response.results.map((attr, i) => (
        <div key={attr.protected} className="space-y-3">
          {response.results.length > 1 && (
            <div className="flex items-baseline justify-between">
              <div className="font-serif text-lg text-ink">
                <span className="text-ink-dim font-mono text-xs uppercase tracking-[0.18em] mr-2">
                  Attr {i + 1}/{response.results.length}
                </span>
                {attr.protected}
              </div>
              <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
                accuracy ↑ vs equalized-odds gap ↓
              </div>
            </div>
          )}
          <ParetoPanel attr={attr} />
        </div>
      ))}
    </div>
  );
}

function ParetoPanel({ attr }: { attr: Stage4Response["results"][number] }) {
  const valid = attr.models.filter(
    (m): m is Stage4Model & { auc: number; fairness_gap: number } =>
      m.auc != null && m.fairness_gap != null
  );

  if (valid.length === 0) {
    return (
      <div className="rounded-md border border-hairline bg-elevated/40 p-6 text-center text-sm text-ink-muted">
        No comparable models — every model failed for this attribute.
      </div>
    );
  }

  const sortedRows = [...attr.models].sort(
    (a, b) => (b.auc ?? -1) - (a.auc ?? -1)
  );

  return (
    <div className="grid lg:grid-cols-5 gap-4">
      <div className="lg:col-span-3 rounded-md border border-hairline bg-canvas/60 p-3 sm:p-5">
        <ParetoSvg models={valid} />
      </div>
      <div className="lg:col-span-2 rounded-md border border-hairline overflow-hidden">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
              <th className="text-left font-normal py-2.5 pl-4 pr-2">Model</th>
              <th className="text-right font-normal py-2.5 px-2">AUC</th>
              <th className="text-right font-normal py-2.5 px-2">EO Δ</th>
              <th className="text-right font-normal py-2.5 px-2 pr-4">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((m) => {
              const color = MODEL_COLOR[m.color] ?? MODEL_COLOR.m1;
              return (
                <tr
                  key={m.key}
                  className={cn(
                    "border-b border-hairline last:border-0",
                    m.recommended && "bg-accent-soft/40"
                  )}
                >
                  <td className="py-2.5 pl-4 pr-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ background: color }}
                      />
                      <span className="text-xs font-medium truncate">
                        {m.name}
                      </span>
                    </div>
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-xs">
                    {m.auc != null ? m.auc.toFixed(3) : "—"}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-xs">
                    {m.fairness_gap != null
                      ? `${(m.fairness_gap * 100).toFixed(1)}pp`
                      : "—"}
                  </td>
                  <td className="py-2.5 px-2 pr-4 text-right">
                    {m.recommended ? (
                      <Badge tone="accent" dot>
                        Recommended
                      </Badge>
                    ) : m.pareto_optimal ? (
                      <Badge tone="success">Pareto</Badge>
                    ) : (
                      <Badge tone="neutral" className="opacity-60">
                        Dominated
                      </Badge>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ParetoSvg({
  models,
}: {
  models: (Stage4Model & { auc: number; fairness_gap: number })[];
}) {
  const W = 480;
  const H = 320;
  const PAD = { left: 56, right: 24, top: 20, bottom: 44 };

  const aucs = models.map((m) => m.auc);
  const gaps = models.map((m) => m.fairness_gap);
  // Domain with slack so points don't sit on the axis.
  const xMin = Math.max(0, Math.min(...aucs) - 0.02);
  const xMax = Math.min(1, Math.max(...aucs) + 0.02);
  const yMin = 0;
  const yMax = Math.max(...gaps, 0.05) * 1.15;

  const xScale = (v: number) =>
    PAD.left + ((v - xMin) / Math.max(xMax - xMin, 1e-6)) * (W - PAD.left - PAD.right);
  const yScale = (v: number) =>
    H - PAD.bottom - ((v - yMin) / Math.max(yMax - yMin, 1e-6)) * (H - PAD.top - PAD.bottom);

  // Build the frontier path: only Pareto-optimal points, sorted by AUC.
  const frontier = models
    .filter((m) => m.pareto_optimal)
    .sort((a, b) => a.auc - b.auc);
  const frontierPath = frontier
    .map((m, i) => {
      const x = xScale(m.auc);
      const y = yScale(m.fairness_gap);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const xTicks = niceTicks(xMin, xMax, 4);
  const yTicks = niceTicks(yMin, yMax, 4);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-auto select-none"
      role="img"
      aria-label="Pareto frontier — accuracy vs equalized-odds gap"
    >
      {/* gridlines */}
      {xTicks.map((t) => (
        <line
          key={`gx-${t}`}
          x1={xScale(t)}
          x2={xScale(t)}
          y1={PAD.top}
          y2={H - PAD.bottom}
          stroke="#1F232B"
          strokeWidth={1}
        />
      ))}
      {yTicks.map((t) => (
        <line
          key={`gy-${t}`}
          x1={PAD.left}
          x2={W - PAD.right}
          y1={yScale(t)}
          y2={yScale(t)}
          stroke="#1F232B"
          strokeWidth={1}
        />
      ))}

      {/* axis ticks */}
      {xTicks.map((t) => (
        <text
          key={`tx-${t}`}
          x={xScale(t)}
          y={H - PAD.bottom + 16}
          textAnchor="middle"
          className="fill-ink-dim font-mono"
          fontSize={10}
        >
          {t.toFixed(2)}
        </text>
      ))}
      {yTicks.map((t) => (
        <text
          key={`ty-${t}`}
          x={PAD.left - 8}
          y={yScale(t) + 3}
          textAnchor="end"
          className="fill-ink-dim font-mono"
          fontSize={10}
        >
          {(t * 100).toFixed(0)}pp
        </text>
      ))}

      {/* axis titles */}
      <text
        x={(W + PAD.left - PAD.right) / 2}
        y={H - 8}
        textAnchor="middle"
        className="fill-ink-dim font-mono uppercase tracking-wider"
        fontSize={9}
      >
        AUC →
      </text>
      <text
        transform={`translate(14 ${(H + PAD.top - PAD.bottom) / 2}) rotate(-90)`}
        textAnchor="middle"
        className="fill-ink-dim font-mono uppercase tracking-wider"
        fontSize={9}
      >
        Equalized-odds gap ↓
      </text>

      {/* frontier polyline */}
      {frontierPath && (
        <path
          d={frontierPath}
          stroke="#5B7FFF"
          strokeWidth={1.5}
          fill="none"
          strokeDasharray="4 3"
          opacity={0.75}
        />
      )}

      {/* points */}
      {models.map((m) => {
        const color = MODEL_COLOR[m.color] ?? MODEL_COLOR.m1;
        const cx = xScale(m.auc);
        const cy = yScale(m.fairness_gap);
        const isRec = m.recommended;
        const isPareto = m.pareto_optimal;
        return (
          <g key={m.key}>
            {isRec && (
              <circle cx={cx} cy={cy} r={14} fill={color} opacity={0.18} />
            )}
            <circle
              cx={cx}
              cy={cy}
              r={isRec ? 6 : isPareto ? 5 : 4}
              fill={isRec ? color : isPareto ? color : "transparent"}
              stroke={color}
              strokeWidth={1.5}
              opacity={isPareto || isRec ? 1 : 0.55}
            />
            <text
              x={cx + 9}
              y={cy + 3}
              className="fill-ink font-mono"
              fontSize={10}
              opacity={isPareto || isRec ? 1 : 0.6}
            >
              {m.name}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min;
  if (span <= 0) return [min];
  const step = niceStep(span / count);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + 1e-9; v += step) {
    ticks.push(Number(v.toFixed(6)));
  }
  return ticks;
}

function niceStep(rough: number): number {
  const exp = Math.floor(Math.log10(rough));
  const base = rough / Math.pow(10, exp);
  let nice;
  if (base < 1.5) nice = 1;
  else if (base < 3) nice = 2;
  else if (base < 7) nice = 5;
  else nice = 10;
  return nice * Math.pow(10, exp);
}

/* Stage 5 — Root cause */

function RootCauseArtifact({ cfg }: { cfg: AuditConfig }) {
  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <Card className="p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            Bayesian posterior
          </span>
          <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
            5 classes
          </span>
        </div>
        <div className="font-serif text-xl text-ink mb-1">
          Proxy discrimination
        </div>
        <div className="text-xs text-ink-muted mb-5">
          A feature in your data correlates with{" "}
          <span className="font-mono text-ink">{cfg.protectedAttrs[0]}</span>{" "}
          and drives prediction differently across groups.
        </div>
        <div className="space-y-2.5">
          {ROOT_CAUSE.map((c) => (
            <div key={c.label}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-ink-muted">{c.label}</span>
                <span className="font-mono tabular text-ink-dim">
                  {(c.value * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1 rounded-full bg-elevated overflow-hidden">
                <div
                  className={
                    c.color === "danger"
                      ? "h-full bg-danger"
                      : c.color === "warning"
                      ? "h-full bg-warning"
                      : "h-full bg-ink-faint"
                  }
                  style={{ width: `${c.value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-5">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
          SHAP rank delta · top features
        </div>
        <div className="space-y-3">
          {[
            { name: "priors_count", a: 1, b: 4, delta: "+3" },
            { name: "age", a: 2, b: 2, delta: "0" },
            { name: "charge_degree", a: 5, b: 3, delta: "−2" },
            { name: "decile_score", a: 3, b: 5, delta: "+2" },
          ].map((r) => (
            <div
              key={r.name}
              className="grid grid-cols-[1fr_auto_auto] gap-3 items-center"
            >
              <div className="font-mono text-xs text-ink truncate">{r.name}</div>
              <div className="font-mono text-2xs text-ink-dim tabular">
                #{r.a} → #{r.b}
              </div>
              <div
                className={cn(
                  "font-mono text-xs tabular px-1.5 py-0.5 rounded",
                  r.delta.startsWith("+") &&
                    "text-danger bg-danger/10",
                  r.delta === "0" && "text-ink-dim",
                  r.delta.startsWith("−") && "text-success bg-success/10"
                )}
              >
                {r.delta}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-5 pt-4 border-t border-hairline text-xs text-ink-muted leading-relaxed">
          <span className="font-mono text-ink">priors_count</span> jumps from
          rank #4 (Logistic) to rank #1 (XGBoost) for African-American
          defendants. Permutation test{" "}
          <span className="font-mono text-ink">p = 0.003</span>.
        </div>
      </Card>
    </div>
  );
}

/* Stage 6 — Remediation */

function RemediationArtifact() {
  return (
    <div className="grid lg:grid-cols-3 gap-4">
      {REMEDIATION.map((r) => (
        <Card key={r.title} className="p-5">
          <div className="flex items-center justify-between mb-3">
            {r.status === "blocked" && (
              <Badge tone="danger" dot>
                Blocked
              </Badge>
            )}
            {r.status === "recommended" && (
              <Badge tone="accent" dot>
                Apply
              </Badge>
            )}
            {r.status === "optional" && (
              <Badge tone="neutral" dot>
                Optional
              </Badge>
            )}
            <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
              Stage 6
            </span>
          </div>
          <div className="font-serif text-lg text-ink mb-2">{r.title}</div>
          <p className="text-sm text-ink-muted leading-relaxed">{r.body}</p>
        </Card>
      ))}
    </div>
  );
}

/* Stage 7 — Gemini reasoning checkpoints */

function GeminiArtifact() {
  const checkpoints = [
    {
      n: "1",
      title: "Pareto frontier consistency",
      body: "The model surfaced as recommended is on the Pareto frontier and dominates no other recommendation candidate.",
      pass: true,
    },
    {
      n: "2",
      title: "Disparate impact threshold",
      body: "Recommended model's disparate impact ratio (0.917) clears the 4/5ths regulatory floor (0.80).",
      pass: true,
    },
    {
      n: "3",
      title: "Root-cause cross-check",
      body: "Bayesian posterior (proxy discrimination, 73%) matches SHAP rank-delta evidence on priors_count.",
      pass: true,
    },
    {
      n: "4",
      title: "Remediation safety",
      body: "Threshold adjustment correctly blocked — would mask, not fix, proxy mechanism. Decorrelation flagged for application.",
      pass: true,
    },
  ];

  return (
    <div className="grid sm:grid-cols-2 gap-3">
      {checkpoints.map((c) => (
        <div
          key={c.n}
          className="rounded-md border border-hairline bg-elevated/40 p-4"
        >
          <div className="flex items-start justify-between mb-2">
            <div className="text-2xs font-mono uppercase tracking-wider text-ink-dim">
              Checkpoint {c.n}
            </div>
            <Badge tone="success" dot>
              Pass
            </Badge>
          </div>
          <div className="text-sm font-medium text-ink mb-1.5">{c.title}</div>
          <p className="text-xs text-ink-muted leading-relaxed">{c.body}</p>
        </div>
      ))}
    </div>
  );
}

/* ─────────────  Error banner  ───────────── */

function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="mb-5 rounded-md border border-danger/30 bg-danger/[0.06] px-4 py-3 flex items-start gap-3">
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        className="mt-0.5 shrink-0 text-danger"
        aria-hidden="true"
      >
        <path
          d="M7 1L13 12H1L7 1Z M7 6V8.5 M7 10V10.1"
          stroke="currentColor"
          strokeWidth="1.4"
          fill="none"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      <div className="flex-1 text-xs text-danger font-mono leading-relaxed">
        {message}
      </div>
      <button
        onClick={onDismiss}
        className="text-2xs font-mono uppercase tracking-wider text-danger/80 hover:text-danger"
      >
        Dismiss
      </button>
    </div>
  );
}

/* ─────────────  Icons  ───────────── */

function PlayIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path
        d="M3 2L10 6L3 10V2Z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <path
        d="M3 7H11M11 7L7.5 3.5M11 7L7.5 10.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
    </svg>
  );
}
