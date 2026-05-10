"use client";

import { useEffect, useRef, useState } from "react";
import { Badge, Button, Card, CardHeader } from "@/components/primitives";
import { cn } from "@/lib/cn";
import {
  runStage1,
  runStage2Stream,
  runStage3Stream,
  runStage4,
  runStage5,
  runStage6,
  runStage7,
  runStage8,
  type Stage1Fingerprint,
  type Stage1Response,
  type Stage2Model,
  type Stage3Response,
  type Stage3Model,
  type Stage3PairwiseTest,
  type Stage4Response,
  type Stage4Model,
  type Stage5Response,
  type Stage5PerAttr,
  type Stage5ProxyFeature,
  type Stage5CorrFeature,
  type Stage6Response,
  type Stage6Action,
  type Stage6ActionStatus,
  type Stage7Response,
  type CP2PerModel,
  type Stage8Response,
  type Stage8DeploymentCondition,
  type DataRemediationAction,
} from "@/lib/api";
import type { ParsedCsv } from "@/lib/csv";
import type { AuditConfig } from "./stage-configure";
import {
  formatGroup,
  attrLabel,
  shortLabel,
  UNKNOWN_MAPPING_HINT,
} from "@/lib/labels";

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

/** Stage 3 progressive view — same shape as Stage3Response, plus a per-model
 *  status flag so the UI can show "running" placeholders before each model's
 *  fairness audit finishes. `complete` flips true on the `done` event. */
type Stage3LiveModel = Stage3Model & { status: "running" | "done" | "error" };
type Stage3Live = {
  session_id: string;
  n_total: number;
  bootstrap_n: number;
  pairwise_tests: Stage3PairwiseTest[];
  results: {
    protected: string;
    groups: string[];
    models: Stage3LiveModel[];
  }[];
  complete: boolean;
};

export type StageData = {
  stage1?: Stage1Response;
  stage2?: Stage2Live;
  stage3?: Stage3Live;
  stage4?: Stage4Response;
  stage5?: Stage5Response;
  stage6?: Stage6Response;
  stage7?: Stage7Response;
  stage8?: Stage8Response;
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
    meta: "BOOTSTRAP · STREAMED · LIVE BACKEND",
    duration: 1200,
    finding: ({ data }) => {
      const r = data.stage3;
      if (!r) return "Fairness metrics computed";
      const attrs = r.results.length;
      const doneModels =
        r.results[0]?.models.filter((m) => m.status === "done").length ?? 0;
      const totalModels = r.results[0]?.models.length ?? 0;
      const eoGaps = r.results.flatMap((x) =>
        x.models
          .filter((m) => m.status === "done")
          .map((m) => m.gaps.eo_gap)
          .filter((v): v is number => v != null)
      );
      const worst = eoGaps.length > 0 ? Math.max(...eoGaps) : null;
      const sigPairs = r.pairwise_tests?.filter((t) => t.significant_auc || t.significant_errors).length ?? 0;
      return `${attrs} attribute${attrs > 1 ? "s" : ""} × ${doneModels}/${totalModels} models · bootstrap n=${r.bootstrap_n}${worst != null ? ` · worst EO Δ ${(worst * 100).toFixed(1)}pp` : ""}${sigPairs > 0 ? ` · ${sigPairs} sig. pairwise tests` : ""}`;
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
        const disqualified = x.models.filter(
          (m) => m.pareto_optimal && !m.fairness_qualified
        ).length;
        const rec = x.models.find((m) => m.recommended);
        return (
          `${x.protected}: ${optimal}/${total} on frontier` +
          (disqualified > 0 ? ` · ${disqualified} disqualified by fairness` : "") +
          (rec ? ` · rec ${rec.name}` : " · no recommendation")
        );
      });
      return `EO threshold ${(r.eo_gap_threshold * 100).toFixed(0)}pp · ${lines.join(" · ")}`;
    },
  },
  {
    id: 5,
    name: "Root cause diagnosis",
    desc: "SHAP feature attribution per group, proxy discrimination detection, counterfactual flip test, and Bayesian 5-class root cause posterior.",
    meta: "SHAP · PERMUTATION · BAYESIAN · LIVE BACKEND",
    duration: 2000,
    finding: ({ data }) => {
      const r = data.stage5;
      if (!r) return "Root cause diagnosed";
      const cause = r.primary_root_cause?.replace(/_/g, " ") ?? "unknown";
      const firstAttr = r.results ? Object.values(r.results)[0] : null;
      const conf = firstAttr?.bayesian_root_cause?.[r.primary_root_cause ?? ""];
      const pct = conf != null ? `${(conf * 100).toFixed(0)}%` : null;
      const flip = firstAttr?.counterfactual_flip_rate;
      return `${cause}${pct ? ` · ${pct} confidence` : ""}${r.shap_available ? " · SHAP validated" : " · correlation-based"}${flip != null ? ` · flip rate ${(flip * 100).toFixed(1)}%` : ""}`;
    },
  },
  {
    id: 6,
    name: "Guided remediation",
    desc: "Root-cause-conditional fixes, validated as Pareto improvements.",
    meta: "CONDITIONAL RULES · LIVE BACKEND",
    duration: 900,
    finding: ({ data }) => {
      const r = data.stage6;
      if (!r) return "Remediation plan generated";
      const rec = r.actions.filter((a) => a.status === "recommended").length;
      const blocked = r.actions.filter((a) => a.status === "blocked").length;
      const safeTxt = r.safe_to_auto_fix ? "safe to apply" : "manual review required";
      return `${r.diagnosis}${blocked > 0 ? ` · ${blocked} intervention${blocked > 1 ? "s" : ""} blocked` : ""}${rec > 0 ? ` · ${rec} recommended` : ""} · ${safeTxt}`;
    },
  },
  {
    id: 7,
    name: "Reasoning & validation layer",
    desc: "Four Pydantic-validated checkpoints cross-check Stages 1–6 for consistency before issuing a final recommendation.",
    meta: "PYDANTIC · 4 CHECKPOINTS · LIVE BACKEND",
    duration: 1100,
    finding: ({ data }) => {
      const r = data.stage7;
      if (!r) return "Reasoning checkpoints complete";
      const rec = r.final_recommendation;
      const verdict = rec.fairness_compliant && rec.pareto_status === "non-dominated"
        ? `recommended ${rec.model}`
        : "no recommendation gated through";
      return `${r.checkpoints_summary} · ${verdict}`;
    },
  },
  {
    id: 8,
    name: "Decision-intelligence report",
    desc: "Five executive-grade tabs: summary · fairness & risk · model behavior · actions · deployment readiness.",
    meta: "5 TABS · LIVE BACKEND",
    duration: 700,
    finding: ({ data }) => {
      const r = data.stage8;
      if (!r) return "Report rendered";
      const v = r.deployment.verdict.replace(/_/g, " ");
      return `${r.deployment.passed_count}/${r.deployment.total_conditions} deployment conditions met · verdict: ${v}`;
    },
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
  onComplete: (totalRunningSec: number, finalData: StageData) => void;
}) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [status, setStatus] = useState<StageStatus>("idle");
  const [stageElapsedMs, setStageElapsedMs] = useState(0);
  const [findings, setFindings] = useState<Record<number, string>>({});
  const [data, setData] = useState<StageData>({});
  const [error, setError] = useState<string | null>(null);
  // Stages the user has already completed at least once — drives clickable
  // navigation back to past artifacts without re-running the backend.
  const [completedIndices, setCompletedIndices] = useState<Set<number>>(new Set());
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
          // Streamed audit — render each model as it completes, mirroring Stage 2.
          let snap: Stage3Live | null = null;
          const minDuration = new Promise<void>((r) =>
            setTimeout(r, stage.duration)
          );
          const stream = runStage3Stream(
            data.stage2.session_id,
            (ev) => {
              if (cancelled) return;
              if (ev.event === "init") {
                snap = {
                  session_id: ev.session_id,
                  n_total: ev.n_total,
                  bootstrap_n: ev.bootstrap_n,
                  pairwise_tests: [],
                  results: ev.results.map((r) => ({
                    protected: r.protected,
                    groups: r.groups,
                    // Placeholder Stage3LiveModel — fields filled in on model_done.
                    models: r.models.map((m) => ({
                      key: m.key as Stage3Model["key"],
                      name: m.name,
                      family: m.family,
                      color: m.color,
                      status: "running" as const,
                      overall_auc: null,
                      overall_auc_ci: [null, null],
                      ece: null,
                      by_group: {},
                      gaps: {
                        tpr_gap: null,
                        fpr_gap: null,
                        eo_gap: null,
                        dp_gap: null,
                        di_ratio: null,
                        ppv_gap: null,
                      },
                    })),
                  })),
                  complete: false,
                };
              } else if (ev.event === "model_done" && snap) {
                if ("error" in ev && ev.error) {
                  snap = {
                    ...snap,
                    results: snap.results.map((r) => ({
                      ...r,
                      models: r.models.map((m) =>
                        m.key === ev.model_key ? { ...m, status: "error" as const } : m
                      ),
                    })),
                  };
                } else if ("model" in ev && ev.model) {
                  const protectedAttr = (ev as { protected: string }).protected;
                  snap = {
                    ...snap,
                    results: snap.results.map((r) =>
                      r.protected === protectedAttr
                        ? {
                            ...r,
                            models: r.models.map((m) =>
                              m.key === ev.model_key
                                ? { ...ev.model, status: "done" as const }
                                : m
                            ),
                          }
                        : r
                    ),
                  };
                }
              } else if (ev.event === "pairwise_done" && snap) {
                snap = { ...snap, pairwise_tests: ev.tests };
              } else if (ev.event === "done" && snap) {
                snap = { ...snap, complete: true };
              }
              if (snap) {
                const next = snap;
                setData((d) => ({ ...d, stage3: next }));
              }
            },
            controller.signal
          );
          await Promise.all([stream, minDuration]);
          if (cancelled) return;
          finalize({ ...data, stage3: snap ?? undefined });
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
        } else if (stage.id === 5) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 5 requires Stage 2 to have completed first");
          }
          // Pass the recommended model key from Stage 4 if available
          const recKey = data.stage4?.results?.[0]?.models?.find(
            (m) => m.recommended
          )?.key;
          const [stage5] = await Promise.all([
            runStage5(data.stage2.session_id, recKey, controller.signal),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage5 }));
          finalize({ ...data, stage5 });
        } else if (stage.id === 6) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 6 requires Stage 2 to have completed first");
          }
          if (!data.stage5) {
            throw new Error("Stage 6 requires Stage 5 to have completed first");
          }
          const [stage6] = await Promise.all([
            runStage6(data.stage2.session_id, data.stage5, controller.signal),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage6 }));
          finalize({ ...data, stage6 });
        } else if (stage.id === 7) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 7 requires earlier stages to have completed first");
          }
          const [stage7] = await Promise.all([
            runStage7(
              data.stage2.session_id,
              {
                stage1: data.stage1,
                stage2: data.stage2,
                stage3: data.stage3,
                stage4: data.stage4,
                stage5: data.stage5,
                stage6: data.stage6,
              },
              controller.signal
            ),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage7 }));
          finalize({ ...data, stage7 });
        } else if (stage.id === 8) {
          if (!data.stage2?.session_id) {
            throw new Error("Stage 8 requires earlier stages to have completed first");
          }
          const [stage8] = await Promise.all([
            runStage8(
              data.stage2.session_id,
              {
                stage1: data.stage1,
                stage4: data.stage4,
                stage5: data.stage5,
                stage6: data.stage6,
                stage7: data.stage7,
              },
              controller.signal
            ),
            new Promise<void>((r) => setTimeout(r, stage.duration)),
          ]);
          if (cancelled) return;
          setData((d) => ({ ...d, stage8 }));
          finalize({ ...data, stage8 });
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
      // Only count execution time on first completion — re-views shouldn't inflate the total.
      if (!completedIndices.has(currentIdx)) {
        totalRunningMsRef.current += stage.duration;
      }
      setCompletedIndices((s) => {
        if (s.has(currentIdx)) return s;
        const next = new Set(s);
        next.add(currentIdx);
        return next;
      });
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
      onComplete(totalRunningMsRef.current / 1000, data);
      return;
    }
    const nextIdx = currentIdx + 1;
    setCurrentIdx(nextIdx);
    setError(null);
    // If the user has already completed this stage on a prior pass, jump straight
    // to its artifact view rather than asking them to re-run it.
    if (completedIndices.has(nextIdx)) {
      setStatus("complete");
      setStageElapsedMs(PIPELINE[nextIdx].duration);
    } else {
      setStatus("idle");
      setStageElapsedMs(0);
    }
  }

  /** Jump back (or forward) to an already-completed stage to re-view its artifact.
   *  Does not re-run the backend — the cached artifact in `data` is what's shown. */
  function jumpToStage(idx: number) {
    if (idx === currentIdx) return;
    if (!completedIndices.has(idx)) return;
    setCurrentIdx(idx);
    setStatus("complete");
    setStageElapsedMs(PIPELINE[idx].duration);
    setError(null);
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
      <PipelineRail
        currentIdx={currentIdx}
        status={status}
        completedIndices={completedIndices}
        onJumpTo={jumpToStage}
      />

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
          {status === "running" &&
          ((stage.id === 2 && data.stage2) || (stage.id === 3 && data.stage3)) ? (
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
        {/* When the user is re-viewing a past stage, offer a quick jump back to
            the furthest stage they've reached (the "live" tip of the pipeline). */}
        {(() => {
          const furthestCompleted =
            completedIndices.size > 0 ? Math.max(...completedIndices) : -1;
          const liveIdx = Math.min(furthestCompleted + 1, PIPELINE.length - 1);
          const isReviewing =
            currentIdx < liveIdx && completedIndices.has(currentIdx);
          if (!isReviewing) return null;
          return (
            <Button
              variant="ghost"
              size="md"
              onClick={() => {
                if (completedIndices.has(liveIdx)) {
                  jumpToStage(liveIdx);
                } else {
                  setCurrentIdx(liveIdx);
                  setStatus("idle");
                  setStageElapsedMs(0);
                  setError(null);
                }
              }}
            >
              ← Jump to current · Stage {liveIdx + 1}
            </Button>
          );
        })()}
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
  completedIndices,
  onJumpTo,
}: {
  currentIdx: number;
  status: StageStatus;
  completedIndices: Set<number>;
  onJumpTo: (idx: number) => void;
}) {
  return (
    <div className="rounded-lg border border-hairline bg-surface/60 px-4 py-4">
      <ol className="grid grid-cols-4 sm:grid-cols-8 gap-2">
        {PIPELINE.map((s, i) => {
          const isCurrent = i === currentIdx;
          let state: "done" | "running" | "current" | "pending";
          if (isCurrent) {
            state =
              status === "running"
                ? "running"
                : status === "complete"
                ? "done"
                : "current";
          } else {
            state = completedIndices.has(i) ? "done" : "pending";
          }
          // Clickable: any completed stage that isn't already the current one.
          const clickable = !isCurrent && completedIndices.has(i);
          const inner = (
            <>
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
            </>
          );
          return (
            <li
              key={s.id}
              className={cn(
                "flex items-center gap-2 min-w-0 rounded-md transition-colors",
                isCurrent && "ring-1 ring-accent/40 bg-accent-soft/20 px-1.5 py-1 -mx-1.5 -my-1"
              )}
            >
              {clickable ? (
                <button
                  type="button"
                  onClick={() => onJumpTo(i)}
                  title={`Re-view Stage ${s.id} — ${s.name}`}
                  className="flex items-center gap-2 min-w-0 w-full rounded-md hover:bg-elevated/70 transition-colors cursor-pointer text-left -mx-1 px-1 py-0.5"
                >
                  {inner}
                </button>
              ) : (
                <div className="flex items-center gap-2 min-w-0 w-full">
                  {inner}
                </div>
              )}
            </li>
          );
        })}
      </ol>
      {/* Hint about navigation, only shown once anything is completed */}
      {completedIndices.size > 0 && (
        <div className="mt-3 text-2xs font-mono text-ink-faint uppercase tracking-wider">
          Tip · click any completed stage to re-view its artifact
        </div>
      )}
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
  // Stage 3 streamed snapshots are also Stage3Response-shaped so downstream
  // stages that read it stay compatible.
  if (stageId === 4) {
    if (!data.stage4) return null;
    return <ParetoArtifact response={data.stage4} />;
  }
  if (stageId === 5) {
    if (!data.stage5) return null;
    return <RootCauseArtifact response={data.stage5} cfg={cfg} />;
  }
  if (stageId === 6) {
    if (!data.stage6) return null;
    return <RemediationArtifact response={data.stage6} />;
  }
  if (stageId === 7) {
    if (!data.stage7) return null;
    return <ReasoningCheckpointsArtifact response={data.stage7} />;
  }
  if (stageId === 8) {
    if (!data.stage8) return null;
    return <ExecutiveReportArtifact response={data.stage8} />;
  }
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
      {/* Data handling & assumptions — surfaced up front so users trust the pipeline */}
      <div className="rounded-md border border-hairline bg-elevated/40 px-4 py-3 text-xs text-ink-muted leading-relaxed">
        <span className="text-ink font-medium">How this audit handles your data: </span>
        <ul className="mt-1.5 ml-1 space-y-0.5">
          <li>
            <span className="text-ink">Classification only.</span>{" "}
            Targets are treated as a binary or categorical label (you set the positive class
            in Step 2). Continuous regression targets aren't supported in this version.
          </li>
          <li>
            <span className="text-ink">Protected attributes are categorical.</span>{" "}
            Numeric columns like <span className="font-mono">age</span> aren't auto-binned —
            convert to a categorical column upstream if you want group-level analysis on it.
          </li>
          <li>
            <span className="text-ink">Missing values.</span>{" "}
            Rows with a missing protected-attribute value are dropped from per-group statistics.
            Missing feature values are imputed by the model's preprocessing pipeline (column median).
          </li>
          <li>
            <span className="text-ink">Small subgroups</span> (n &lt; {response.results[0]?.fingerprint.n_threshold ?? 100})
            are flagged in yellow — bootstrap confidence intervals on those groups will be unreliable.
          </li>
        </ul>
      </div>
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
  const pp = result.preprocessing;

  return (
    <>
      {/* Preprocessing transparency — surfaces how many rows were dropped
          for missing target or missing protected attribute. Shown only when
          something was actually dropped, so happy paths stay clean. */}
      {pp && pp.n_dropped_total > 0 && (
        <div
          className={cn(
            "rounded-md border px-4 py-3 text-xs leading-relaxed mb-3",
            (pp.drop_rate ?? 0) > 0.10
              ? "border-warning/30 bg-warning/[0.06] text-warning"
              : "border-hairline bg-elevated/30 text-ink-muted"
          )}
        >
          <span className="font-medium">Preprocessing transparency · {protectedName}: </span>
          {pp.n_dropped_total.toLocaleString()} of{" "}
          <span className="font-mono">{pp.n_input.toLocaleString()}</span> rows
          dropped before subgroup analysis
          {pp.drop_rate != null && (
            <span className="font-mono">
              {" "}({(pp.drop_rate * 100).toFixed(1)}%)
            </span>
          )}
          .{" "}
          {pp.n_dropped_target_missing > 0 && (
            <>
              <span className="font-mono">{pp.n_dropped_target_missing}</span>{" "}
              had a missing target value;{" "}
            </>
          )}
          {pp.n_dropped_protected_missing > 0 && (
            <>
              <span className="font-mono">{pp.n_dropped_protected_missing}</span>{" "}
              had a missing <span className="font-mono">{protectedName}</span> value
              {(pp.drop_rate ?? 0) > 0.10 && (
                <> — this is a meaningful share; consider whether the missingness is itself a fairness signal</>
              )}
              .
            </>
          )}
        </div>
      )}
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
            const fmt = formatGroup(protectedName, g.name);
            return (
              <div key={g.name}>
                <div className="flex items-baseline justify-between text-xs mb-1">
                  <span className="text-ink" title={fmt.full}>
                    {fmt.display ?? g.name}
                    {fmt.display && (
                      <span className="ml-1.5 text-ink-faint font-mono text-2xs">
                        ({protectedName} = {g.name})
                      </span>
                    )}
                    {!fmt.known && (
                      <span
                        className="ml-1.5 text-ink-faint text-2xs"
                        title={UNKNOWN_MAPPING_HINT}
                      >
                        ⓘ
                      </span>
                    )}
                  </span>
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
                const fmt = formatGroup(protectedName, g.name);
                return (
                  <div key={g.name}>
                    <div className="flex items-baseline justify-between text-xs mb-1">
                      <span className="text-ink-muted" title={fmt.full}>
                        {fmt.display ?? g.name}
                      </span>
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
    </>
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

function FairnessArtifact({ response }: { response: Stage3Live }) {
  const totalModels = response.results[0]?.models.length ?? 0;
  const doneModels =
    response.results[0]?.models.filter((m) => m.status === "done").length ?? 0;
  return (
    <div className="space-y-8">
      {/* Stream progress chip — shown while models are still auditing */}
      {!response.complete && (
        <div className="flex items-center gap-2 text-2xs font-mono uppercase tracking-wider text-ink-dim">
          <span className="relative w-2 h-2 inline-block">
            <span className="absolute inset-0 rounded-full bg-accent/30 animate-ping" />
            <span className="absolute inset-0 rounded-full bg-accent" />
          </span>
          {doneModels}/{totalModels} models audited · bootstrap n={response.bootstrap_n.toLocaleString()}
        </div>
      )}
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
      {response.pairwise_tests && response.pairwise_tests.length > 0 && (
        <PairwiseTestsTable
          tests={response.pairwise_tests}
          predictions={response.results[0]?.models ?? []}
        />
      )}
    </div>
  );
}

function FairnessTable({ attr }: { attr: Stage3Live["results"][number] }) {
  // Sort: completed models first by AUC, running models last in their original order.
  const sorted = [...attr.models].sort((a, b) => {
    if (a.status === "running" && b.status !== "running") return 1;
    if (b.status === "running" && a.status !== "running") return -1;
    return (b.overall_auc ?? 0) - (a.overall_auc ?? 0);
  });
  const groups = attr.groups;
  const protectedName = attr.protected;

  return (
    <div className="overflow-x-auto rounded-md border border-hairline">
      <table className="w-full text-sm tabular">
        <thead>
          <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
            <th className="text-left font-normal py-3 pl-4 pr-3">Model</th>
            <th className="text-right font-normal py-3 px-3">AUC overall</th>
            {groups.map((g) => {
              const fmt = formatGroup(protectedName, g);
              const display = shortLabel(protectedName, g);
              return (
                <th
                  key={g}
                  className="text-right font-normal py-3 px-3"
                  title={`AUC for ${fmt.full}`}
                >
                  AUC · {display}
                  {fmt.known && (
                    <div className="text-ink-faint text-2xs font-mono normal-case tracking-normal">
                      {protectedName}={g}
                    </div>
                  )}
                </th>
              );
            })}
            <th className="text-right font-normal py-3 px-3" title="max−min TPR across groups">TPR Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max−min FPR across groups">FPR Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max(TPR Δ, FPR Δ)">EO Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max−min selection rate">DP Δ</th>
            <th className="text-right font-normal py-3 px-3" title="max−min PPV (precision) across groups">PPV Δ</th>
            <th className="text-right font-normal py-3 px-3" title="min/max selection rate; 4/5 rule = 0.80">DI ratio</th>
            <th className="text-right font-normal py-3 px-3 pr-4" title="Expected Calibration Error — lower is better">ECE</th>
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
  model: Stage3LiveModel;
  groups: string[];
}) {
  const color = MODEL_COLOR[model.color] ?? MODEL_COLOR.m1;
  const eo = model.gaps.eo_gap;
  const dp = model.gaps.dp_gap;
  const tprΔ = model.gaps.tpr_gap;
  const fprΔ = model.gaps.fpr_gap;
  const di = model.gaps.di_ratio;
  const degeneracy = degenerateClassifier(model, groups);
  const isRunning = model.status === "running";
  const isError = model.status === "error";

  // Streaming: render a slim placeholder row while the model's audit is in flight.
  if (isRunning || isError) {
    const span = 8 + groups.length; // total columns in the table
    return (
      <tr className="border-b border-hairline last:border-0">
        <td className="py-3 pl-4 pr-3">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ background: color }}
            />
            <div className="min-w-0">
              <span className="font-medium truncate">{model.name}</span>
              <div className="text-2xs text-ink-dim font-mono truncate">
                {model.family}
              </div>
            </div>
          </div>
        </td>
        <td colSpan={span} className="py-3 px-3 text-2xs font-mono uppercase tracking-wider">
          {isRunning ? (
            <span className="inline-flex items-center gap-2 text-ink-dim">
              <span className="relative w-2 h-2 inline-block">
                <span className="absolute inset-0 rounded-full bg-accent/30 animate-ping" />
                <span className="absolute inset-0 rounded-full bg-accent" />
              </span>
              auditing fairness · bootstrap CIs running…
            </span>
          ) : (
            <span className="text-danger">audit failed</span>
          )}
        </td>
      </tr>
    );
  }

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
      <td className="py-3 px-3 text-right font-mono">
        <GapCell value={model.gaps.ppv_gap} threshold={0.1} />
      </td>
      <td className="py-3 px-3 text-right font-mono">
        {di == null ? (
          <span className="text-ink-dim">—</span>
        ) : (
          <span className={cn(di < 0.8 ? "text-warning" : "text-ink")}>
            {di.toFixed(2)}
          </span>
        )}
      </td>
      <td className="py-3 px-3 pr-4 text-right font-mono">
        {model.ece == null ? (
          <span className="text-ink-dim">—</span>
        ) : (
          <span className={cn(model.ece > 0.1 ? "text-warning" : "text-ink")}>
            {model.ece.toFixed(3)}
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

/* Pairwise statistical tests table — McNemar (error disagreement) + DeLong (AUC). */

const MODEL_NAME_MAP = (models: Stage3Model[], key: string) =>
  models.find((m) => m.key === key)?.name ?? key;

function PairwiseTestsTable({
  tests,
  predictions,
}: {
  tests: Stage3PairwiseTest[];
  predictions: Stage3Model[];
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Pairwise model tests
        </div>
        <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
          BH FDR q = 0.05 · McNemar (errors) · DeLong (AUC)
        </div>
      </div>
      <div className="overflow-x-auto rounded-md border border-hairline">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
              <th className="text-left font-normal py-2.5 pl-4 pr-3">Model A</th>
              <th className="text-left font-normal py-2.5 px-3">Model B</th>
              <th className="text-right font-normal py-2.5 px-3" title="McNemar p-value (raw)">McNemar p</th>
              <th className="text-right font-normal py-2.5 px-3" title="McNemar p after BH FDR correction">McNemar p adj</th>
              <th className="text-right font-normal py-2.5 px-3" title="DeLong AUC difference p-value (raw)">DeLong p</th>
              <th className="text-right font-normal py-2.5 px-3 pr-4" title="DeLong p after BH FDR correction">DeLong p adj</th>
            </tr>
          </thead>
          <tbody>
            {tests.map((t) => {
              const nameA = MODEL_NAME_MAP(predictions, t.model_a);
              const nameB = MODEL_NAME_MAP(predictions, t.model_b);
              return (
                <tr
                  key={`${t.model_a}-${t.model_b}`}
                  className={cn(
                    "border-b border-hairline last:border-0",
                    (t.significant_errors || t.significant_auc) && "bg-warning/[0.04]"
                  )}
                >
                  <td className="py-2.5 pl-4 pr-3 text-xs font-medium text-ink">{nameA}</td>
                  <td className="py-2.5 px-3 text-xs font-medium text-ink">{nameB}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-xs">
                    <PvalCell value={t.mcnemar_p} />
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-xs">
                    <PvalCell value={t.mcnemar_p_adj} significant={t.significant_errors} />
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-xs">
                    <PvalCell value={t.delong_p} />
                  </td>
                  <td className="py-2.5 px-3 pr-4 text-right font-mono text-xs">
                    <PvalCell value={t.delong_p_adj} significant={t.significant_auc} />
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

function PvalCell({ value, significant }: { value: number | null; significant?: boolean }) {
  if (value == null) return <span className="text-ink-dim">—</span>;
  const cls =
    significant ? "text-warning font-medium" :
    value < 0.05 ? "text-warning" :
    "text-ink";
  return (
    <span className={cls}>
      {value < 0.001 ? "<0.001" : value.toFixed(3)}
      {significant && <span className="ml-1 text-2xs">*</span>}
    </span>
  );
}

/* Stage 4 — Pareto frontier per protected attribute. AUC (X) vs equalized
   odds gap (Y, lower=fairer). Dominated models are dimmed; the recommended
   pick is highlighted. */

function ParetoArtifact({ response }: { response: Stage4Response }) {
  const thresholdPct = (response.eo_gap_threshold * 100).toFixed(0);
  return (
    <div className="space-y-8">
      {/* Decision-rule explainer — three-step rule with threshold guardrail */}
      <div className="rounded-md border border-accent/20 bg-accent-soft/30 px-5 py-4 text-xs text-ink leading-relaxed">
        <div className="font-mono uppercase tracking-[0.18em] text-2xs text-accent mb-2">
          How a model is chosen
        </div>
        <ol className="space-y-1.5 list-decimal list-inside marker:text-accent marker:font-mono">
          <li>
            <span className="text-ink font-medium">Pareto filter.</span>{" "}
            Drop any model that's beaten on{" "}
            <em className="not-italic text-ink">both</em> accuracy (AUC) and
            fairness (equalized-odds gap) by some other model — those are{" "}
            <span className="text-danger font-medium">dominated</span> and can't be
            optimal under any preference.
          </li>
          <li>
            <span className="text-ink font-medium">Fairness threshold.</span>{" "}
            Disqualify any survivor whose EO gap exceeds{" "}
            <span className="font-mono">{thresholdPct}pp</span> — fairness is a
            constraint here, not just a tiebreaker. A high-AUC model that fails
            this guardrail is{" "}
            <span className="text-warning font-medium">not</span> recommended.
          </li>
          <li>
            <span className="text-ink font-medium">Reject degenerate models.</span>{" "}
            A model that predicts a single class for every input has trivially zero
            gaps and would slip through the previous filters — it's labeled{" "}
            <span className="text-danger font-medium">Degenerate</span> and refused.
          </li>
          <li>
            <span className="text-ink font-medium">Highest accuracy wins.</span>{" "}
            Among models that pass all three filters, pick the one with the highest
            AUC. That's the{" "}
            <span className="text-success font-medium">recommended</span> model.
          </li>
        </ol>
        <div className="mt-2.5 text-2xs text-ink-muted font-mono">
          Composite ranking inside the qualifying set: <span className="text-ink">Score = AUC − {response.lambda_param.toFixed(1)}·EO_gap</span> · higher is better.
        </div>
      </div>
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
          <ParetoPanel attr={attr} eoThreshold={response.eo_gap_threshold} />
          <ParetoTradeoffPanel
            attr={attr}
            eoThreshold={response.eo_gap_threshold}
          />
          {/* Surfaced when no model satisfies the fairness threshold —
              keeps the UI honest about why nothing was recommended. */}
          {attr.recommendation_warning && (
            <div className="rounded-md border border-warning/30 bg-warning/[0.06] px-4 py-3 text-xs text-warning leading-relaxed">
              <div className="font-mono uppercase tracking-[0.18em] text-2xs mb-1.5">
                ⚠ No recommendation issued
              </div>
              {attr.recommendation_warning}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Plain-English explainer of why this specific model was recommended,
 *  computed by comparing the recommended model to its closest non-recommended
 *  alternative. Surfaces the actual tradeoff and the threshold guardrail. */
function ParetoTradeoffPanel({
  attr,
  eoThreshold,
}: {
  attr: Stage4Response["results"][number];
  eoThreshold: number;
}) {
  const rec = attr.models.find((m) => m.recommended);
  if (!rec || rec.auc == null || rec.fairness_gap == null) return null;

  // Highest-AUC model that's NOT recommended — could be a Pareto-optimal model
  // disqualified by the fairness threshold, or a dominated higher-AUC model.
  const competitor = attr.models
    .filter((m) => !m.recommended && m.auc != null && m.fairness_gap != null)
    .sort((a, b) => (b.auc ?? 0) - (a.auc ?? 0))[0];

  const aucDelta = competitor && competitor.auc != null
    ? (rec.auc - competitor.auc) * 100
    : null;
  const gapDelta = competitor && competitor.fairness_gap != null
    ? (rec.fairness_gap - competitor.fairness_gap) * 100
    : null;

  // If the runner-up has higher AUC but is over the fairness threshold, the
  // tradeoff explanation should highlight that the threshold disqualified it —
  // this is the single most important narrative the user should see.
  const competitorOverThreshold =
    competitor != null
      && competitor.fairness_gap != null
      && competitor.fairness_gap > eoThreshold
      && (competitor.auc ?? 0) > rec.auc;

  return (
    <div className="rounded-md border border-success/20 bg-success/[0.04] px-4 py-3 text-xs text-ink leading-relaxed">
      <div className="font-mono uppercase tracking-[0.18em] text-2xs text-success mb-1.5">
        Why <span className="text-ink">{rec.name}</span> was recommended
      </div>
      <div className="text-ink-muted space-y-1.5">
        <div>
          {attr.recommended_reason ?? (
            <>
              Highest AUC <span className="font-mono text-ink">{rec.auc.toFixed(3)}</span>{" "}
              on the Pareto frontier with EO gap{" "}
              <span className="font-mono text-ink">{(rec.fairness_gap * 100).toFixed(1)}pp</span>{" "}
              (within {(eoThreshold * 100).toFixed(0)}pp threshold).
            </>
          )}
          {rec.composite_score != null && (
            <span className="ml-1 text-ink-faint">
              · composite score <span className="font-mono">{rec.composite_score.toFixed(3)}</span>
            </span>
          )}
        </div>
        {competitorOverThreshold && competitor && competitor.fairness_gap != null && (
          <div className="text-warning">
            ⚠ <span className="text-ink">{competitor.name}</span> has higher AUC{" "}
            <span className="font-mono">{(competitor.auc as number).toFixed(3)}</span>{" "}
            but its EO gap{" "}
            <span className="font-mono">{(competitor.fairness_gap * 100).toFixed(1)}pp</span>{" "}
            exceeds the {(eoThreshold * 100).toFixed(0)}pp fairness threshold —
            disqualified from being recommended.
          </div>
        )}
        {!competitorOverThreshold && competitor && aucDelta != null && gapDelta != null && (
          <div>
            Trades{" "}
            <span className={cn("font-mono", aucDelta >= 0 ? "text-success" : "text-warning")}>
              {aucDelta >= 0 ? "+" : ""}
              {aucDelta.toFixed(1)}pp AUC
            </span>{" "}
            for{" "}
            <span className={cn("font-mono", gapDelta <= 0 ? "text-success" : "text-warning")}>
              {gapDelta >= 0 ? "+" : ""}
              {gapDelta.toFixed(1)}pp gap
            </span>{" "}
            vs. <span className="text-ink">{competitor.name}</span>.
          </div>
        )}
      </div>
    </div>
  );
}

function ParetoPanel({
  attr,
  eoThreshold,
}: {
  attr: Stage4Response["results"][number];
  eoThreshold: number;
}) {
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
        <ParetoSvg models={valid} eoThreshold={eoThreshold} />
      </div>
      <div className="lg:col-span-2 rounded-md border border-hairline overflow-hidden">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
              <th className="text-left font-normal py-2.5 pl-4 pr-2">Model</th>
              <th className="text-right font-normal py-2.5 px-2">AUC</th>
              <th className="text-right font-normal py-2.5 px-2" title="Equalized Odds gap">EO Δ</th>
              <th className="text-right font-normal py-2.5 px-2" title="Predictive Parity gap">PPV Δ</th>
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
                  <td className="py-2.5 px-2 text-right font-mono text-xs">
                    {m.ppv_gap != null
                      ? `${(m.ppv_gap * 100).toFixed(1)}pp`
                      : "—"}
                  </td>
                  <td className="py-2.5 px-2 pr-4 text-right">
                    {m.degenerate ? (
                      <span title={`Model predicts ${m.degenerate === "all-negative" ? "all 0" : "all 1"} for every input. Gaps are trivially zero but the model is decision-theoretically useless. Stage 4 explicitly refuses to recommend it.`}>
                        <Badge tone="danger" dot>
                          Degenerate
                        </Badge>
                      </span>
                    ) : m.recommended ? (
                      <span title="Highest AUC among models that are Pareto-optimal AND satisfy the fairness threshold AND are not degenerate. The system's pick.">
                        <Badge tone="success" dot>
                          Recommended
                        </Badge>
                      </span>
                    ) : m.pareto_optimal && !m.fairness_qualified ? (
                      <span title="Pareto-optimal but disqualified because its EO gap exceeds the fairness threshold. Fairness is enforced as a constraint, not a tiebreaker.">
                        <Badge tone="warning" className="border-dashed">
                          Disqualified
                        </Badge>
                      </span>
                    ) : m.pareto_optimal ? (
                      <span title="Pareto-optimal and within the fairness threshold — a viable alternative if you want a different accuracy/fairness tradeoff than the recommended pick.">
                        <Badge tone="warning">Pareto</Badge>
                      </span>
                    ) : (
                      <span title="Dominated: at least one other model is both more accurate AND fairer. There's no reason to pick this one.">
                        <Badge tone="danger" className="opacity-70">
                          Dominated
                        </Badge>
                      </span>
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
  eoThreshold,
}: {
  models: (Stage4Model & { auc: number; fairness_gap: number })[];
  eoThreshold: number;
}) {
  const W = 480;
  const H = 320;
  const PAD = { left: 56, right: 24, top: 20, bottom: 44 };

  const aucs = models.map((m) => m.auc);
  const gaps = models.map((m) => m.fairness_gap);
  // Domain with slack so points don't sit on the axis. Always include the
  // threshold so the user can see whether models cluster above or below it.
  const xMin = Math.max(0, Math.min(...aucs) - 0.02);
  const xMax = Math.min(1, Math.max(...aucs) + 0.02);
  const yMin = 0;
  const yMax = Math.max(...gaps, eoThreshold * 1.2, 0.05) * 1.15;

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

      {/* Fairness threshold band — shades the disqualified region above the
          guardrail and draws a dashed line at the threshold value. */}
      {eoThreshold <= yMax && (
        <>
          <rect
            x={PAD.left}
            y={PAD.top}
            width={W - PAD.left - PAD.right}
            height={Math.max(0, yScale(eoThreshold) - PAD.top)}
            fill="#FBBF24"
            opacity={0.07}
          />
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={yScale(eoThreshold)}
            y2={yScale(eoThreshold)}
            stroke="#FBBF24"
            strokeWidth={1.2}
            strokeDasharray="3 3"
            opacity={0.7}
          />
          <text
            x={W - PAD.right - 4}
            y={yScale(eoThreshold) - 4}
            textAnchor="end"
            className="fill-warning font-mono"
            fontSize={9}
          >
            fairness threshold {(eoThreshold * 100).toFixed(0)}pp
          </text>
        </>
      )}

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

      {/* points — outer halo color matches the verdict so the chart visually
          matches the table; inner fill stays the model color */}
      {models.map((m) => {
        const color = MODEL_COLOR[m.color] ?? MODEL_COLOR.m1;
        const cx = xScale(m.auc);
        const cy = yScale(m.fairness_gap);
        const isRec = m.recommended;
        const isPareto = m.pareto_optimal;
        const isDisqualified = isPareto && !m.fairness_qualified;
        // Verdict ring color (CSS-named so it picks up theme tokens).
        const ringColor = isRec
          ? "#34D399"   // success / green
          : isDisqualified
          ? "#FB923C"   // orange — Pareto but over fairness threshold
          : isPareto
          ? "#FBBF24"   // warning / yellow
          : "#F87171";  // danger / red
        return (
          <g key={m.key}>
            {isRec && (
              <circle cx={cx} cy={cy} r={14} fill={ringColor} opacity={0.22} />
            )}
            <circle
              cx={cx}
              cy={cy}
              r={isRec ? 6 : isPareto ? 5 : 4}
              fill={isRec || (isPareto && !isDisqualified) ? color : "transparent"}
              stroke={ringColor}
              strokeWidth={2}
              strokeDasharray={isDisqualified ? "3 2" : undefined}
              opacity={isPareto || isRec ? 1 : 0.65}
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
      {/* Verdict legend — four-state to match the table */}
      <g transform={`translate(${PAD.left + 6} ${PAD.top + 6})`}>
        <circle cx={0} cy={4} r={4} fill="transparent" stroke="#34D399" strokeWidth={2} />
        <text x={9} y={7} fontSize={9} className="fill-ink-dim font-mono">Recommended</text>
        <circle cx={92} cy={4} r={4} fill="transparent" stroke="#FBBF24" strokeWidth={2} />
        <text x={101} y={7} fontSize={9} className="fill-ink-dim font-mono">Pareto</text>
        <circle
          cx={148}
          cy={4}
          r={4}
          fill="transparent"
          stroke="#FB923C"
          strokeWidth={2}
          strokeDasharray="3 2"
        />
        <text x={157} y={7} fontSize={9} className="fill-ink-dim font-mono">Disqualified</text>
        <circle cx={236} cy={4} r={4} fill="transparent" stroke="#F87171" strokeWidth={2} />
        <text x={245} y={7} fontSize={9} className="fill-ink-dim font-mono">Dominated</text>
      </g>
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

const CAUSE_LABELS: Record<string, string> = {
  proxy_discrimination: "Proxy discrimination",
  representation_bias: "Representation bias",
  label_bias: "Label bias",
  threshold_effect: "Threshold effect",
  model_complexity_bias: "Model complexity bias",
};

/** Technical, single-sentence definition — kept for the existing slot. */
const CAUSE_DESC: Record<string, string> = {
  proxy_discrimination: "A feature correlated with the protected attribute drives predictions differently across groups.",
  representation_bias: "Group underrepresentation skews learned boundaries and evaluation metrics.",
  label_bias: "Historical decisions embedded in labels propagate bias into predictions.",
  threshold_effect: "Decision-boundary placement produces disparate outcomes even from fair scores.",
  model_complexity_bias: "Model over-fits group-specific noise patterns not present at deployment.",
};

/** Plain-English version — speaks to non-ML readers without jargon. */
const CAUSE_PLAIN: Record<string, string> = {
  proxy_discrimination:
    "Even though the protected attribute itself wasn't shown to the model, another feature is quietly acting as a stand-in for it. The model is taking a back door.",
  representation_bias:
    "One group has too few examples in the training data, so the model learned mostly from the other group and doesn't make fair predictions for the smaller one.",
  label_bias:
    "The 'right answers' the model learned from already reflect bias in past human decisions. Fixing the model alone can't undo biased history.",
  threshold_effect:
    "The model's raw scores are roughly fair across groups, but the cutoff line that decides 'yes/no' falls in a place that affects groups unequally.",
  model_complexity_bias:
    "The model is too elaborate and is picking up on group-specific quirks in the training data that won't hold up in the real world.",
};

/** One-line plain-English version of each Stage-5 metric for inline help text. */
const METRIC_HELP = {
  flipRate:
    "If we made a person look like they belong to the other group (by changing the most group-correlated features), how often does the model's decision flip? Higher = the model is leaning hard on group-related features.",
  permTest:
    "Did the difference between groups happen by chance, or is the pattern real? p < 0.05 means the pattern is unlikely to be random.",
  baseRateGap:
    "How different the actual positive rates are between the two groups in the data itself — before any model gets involved.",
} as const;

/** Build a forensic, dataset-specific plain-English explanation for the
 *  diagnosed root cause. Names the actual minority/majority groups (with
 *  their human labels and sample counts) and ties to model behavior. */
function buildForensicSummary(
  cause: string,
  attrName: string,
  groupSizes: Record<string, number>,
  groupPosRates: Record<string, number | null>,
  modelName: string,
  flipRate: number | null,
  baseRateGap: number | null,
  topProxyFeature: string | null
): string {
  const sized = Object.entries(groupSizes).sort((a, b) => a[1] - b[1]);
  if (sized.length < 2) return "";
  const [minRaw, minN] = sized[0];
  const [majRaw, majN] = sized[sized.length - 1];
  const minFmt = formatGroup(attrName, minRaw);
  const majFmt = formatGroup(attrName, majRaw);
  const minName = minFmt.display ?? minRaw;
  const majName = majFmt.display ?? majRaw;
  const ratio = majN / Math.max(minN, 1);
  const attrPlain = attrLabel(attrName);

  const minPos = groupPosRates[minRaw];
  const majPos = groupPosRates[majRaw];
  const posSummary =
    minPos != null && majPos != null
      ? ` Positive rate: ${(minPos * 100).toFixed(0)}% for ${minName} vs. ${(majPos * 100).toFixed(0)}% for ${majName}.`
      : "";

  switch (cause) {
    case "representation_bias":
      return (
        `${minName} (${attrName} = ${minRaw}, n = ${minN}) is underrepresented compared to ` +
        `${majName} (${attrName} = ${majRaw}, n = ${majN}) — ${ratio.toFixed(1)}× more samples ` +
        `in the majority group. ${modelName} learned its decision boundaries primarily ` +
        `from the larger group, so it doesn't generalize fairly to ${minName}.`
      );
    case "proxy_discrimination":
      return (
        `${modelName} is using${topProxyFeature ? ` the feature "${topProxyFeature}"` : " a feature"} ` +
        `as a backdoor signal for ${attrPlain}. Even though ${attrName} itself wasn't a model input, ` +
        `swapping a person's most correlated features changes the prediction ` +
        `${flipRate != null ? `${(flipRate * 100).toFixed(0)}% of the time` : "noticeably often"} — ` +
        `evidence the model is treating ${minName} (${attrName} = ${minRaw}) and ${majName} ` +
        `(${attrName} = ${majRaw}) differently through that proxy.`
      );
    case "label_bias":
      return (
        `The training labels themselves differ across groups: positive rate is ${(minPos != null ? minPos * 100 : 0).toFixed(0)}% for ` +
        `${minName} (n = ${minN}) vs. ${(majPos != null ? majPos * 100 : 0).toFixed(0)}% for ${majName} (n = ${majN}). ` +
        `Because flipping group-correlated features barely changes predictions, the bias lives in the ` +
        `ground truth, not the features — ${modelName} is faithfully reproducing the historical decisions it was given.`
      );
    case "threshold_effect":
      return (
        `${modelName}'s raw scores look comparable across groups, but the 0.5 cutoff lands in a different ` +
        `place for each one — producing different selection rates for ${minName} (${attrName} = ${minRaw}) ` +
        `vs. ${majName} (${attrName} = ${majRaw}).${posSummary} The fix is per-group threshold optimization, ` +
        `not retraining.`
      );
    case "model_complexity_bias":
      return (
        `${modelName} is overfitting group-specific quirks. ${minName} (n = ${minN}) and ${majName} ` +
        `(n = ${majN}) have noisy patterns the model has memorized rather than generalized — a simpler ` +
        `or more regularized model would treat them more consistently.`
      );
    default:
      return "";
  }
}

function RootCauseArtifact({ response, cfg }: { response: Stage5Response; cfg: AuditConfig }) {
  const firstAttr = response.results ? Object.values(response.results)[0] : null;
  const attrName  = response.results ? Object.keys(response.results)[0] : cfg.protectedAttrs[0];

  if (!firstAttr) {
    return (
      <div className="rounded-md border border-hairline bg-elevated/40 p-6 text-center text-sm text-ink-muted">
        No protected attributes with ≥2 groups found.
      </div>
    );
  }

  const posterior = firstAttr.bayesian_root_cause;
  const primaryCause = response.primary_root_cause ?? "";
  const sorted = Object.entries(posterior).sort(([, a], [, b]) => (b ?? 0) - (a ?? 0));
  const primaryConfidence = posterior[primaryCause] ?? 0;

  // Forensic specifics — grounded in the actual dataset (group names, counts).
  const forensic = buildForensicSummary(
    primaryCause,
    attrName,
    firstAttr.group_sizes ?? {},
    firstAttr.group_positive_rates ?? {},
    response.model_name,
    firstAttr.counterfactual_flip_rate,
    firstAttr.base_rate_gap,
    firstAttr.proxy_features[0]?.feature ?? null
  );

  return (
    <div className="space-y-5">
      {/* Plain-English summary — forensic, grounded in this dataset's groups */}
      <div className="rounded-md border border-accent/20 bg-accent-soft/30 px-5 py-4">
        <div className="flex items-baseline gap-2 mb-1.5">
          <span className="text-2xs font-mono uppercase tracking-[0.18em] text-accent">
            In plain English
          </span>
          <span className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
            why is the model unfair?
          </span>
        </div>
        <p className="text-sm text-ink leading-relaxed mb-2">
          The leading explanation is{" "}
          <span className="text-ink font-medium">
            {(CAUSE_LABELS[primaryCause] ?? primaryCause).toLowerCase()}
          </span>{" "}
          — about{" "}
          <span className="font-mono">
            {(primaryConfidence * 100).toFixed(0)}%
          </span>{" "}
          confident.
        </p>
        {forensic && (
          <p className="text-sm text-ink-muted leading-relaxed">{forensic}</p>
        )}
      </div>

      {/* Group sample-count table — surface the actual numbers driving the diagnosis */}
      <GroupContextPanel
        attrName={attrName}
        sizes={firstAttr.group_sizes ?? {}}
        posRates={firstAttr.group_positive_rates ?? {}}
      />

      {/* Header row: verdict + stats */}
      <div className="grid lg:grid-cols-3 gap-5">
        {/* Bayesian posterior — real inference: discrete prior over 5 cause
            classes × independent Gaussian likelihoods over standardized
            observables, posterior via exact Bayes' rule, credible intervals
            from Monte Carlo over observable bootstrap noise (n=2000). */}
        <Card className="lg:col-span-2 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Bayesian posterior · 5 classes
              </div>
              <div className="text-2xs text-ink-faint mt-0.5">
                Prior × Gaussian likelihood, MC-sampled (n=2,000) for credible intervals
              </div>
            </div>
            <div className="flex items-center gap-2">
              {!response.shap_available && (
                <span title="Computed from feature–group correlations only (the SHAP library wasn't available)">
                  <Badge tone="neutral">correlation-based</Badge>
                </span>
              )}
              {response.shap_available && (
                <span title="Validated using SHAP — measures how much each feature contributes to each prediction">
                  <Badge tone="success" dot>SHAP validated</Badge>
                </span>
              )}
            </div>
          </div>
          <div className="font-serif text-xl text-ink mb-1">
            {CAUSE_LABELS[primaryCause] ?? primaryCause}
          </div>
          <div className="text-xs text-ink-muted mb-1 max-w-lg">
            {CAUSE_DESC[primaryCause] ?? ""}
          </div>
          <div className="text-xs text-ink-faint mb-5 max-w-lg">
            Protected attribute:{" "}
            <span className="font-mono text-ink">{attrName}</span>
            {" · "}model:{" "}
            <span className="font-mono text-ink">{response.model_name}</span>
          </div>
          <div className="space-y-3">
            {sorted.map(([key, val]) => {
              const pct = (val ?? 0) * 100;
              const isPrimary = key === primaryCause;
              const full = firstAttr.bayesian_root_cause_full?.[key];
              const ciLo = full?.ci_low != null ? full.ci_low * 100 : null;
              const ciHi = full?.ci_high != null ? full.ci_high * 100 : null;
              const prior = full?.prior;
              return (
                <div key={key} title={CAUSE_PLAIN[key] ?? ""}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className={cn("text-ink-muted", isPrimary && "text-ink font-medium")}>
                      {CAUSE_LABELS[key] ?? key}
                      {prior != null && (
                        <span className="ml-1.5 text-2xs font-mono text-ink-faint">
                          prior {(prior * 100).toFixed(0)}%
                        </span>
                      )}
                    </span>
                    <span className="font-mono tabular text-ink-dim">
                      {pct.toFixed(0)}%
                      {ciLo != null && ciHi != null && (
                        <span className="ml-1.5 text-2xs text-ink-faint">
                          [{ciLo.toFixed(0)}–{ciHi.toFixed(0)}]
                        </span>
                      )}
                    </span>
                  </div>
                  {/* Bar with embedded 95% credible-interval whisker */}
                  <div className="relative h-1.5 rounded-full bg-elevated overflow-hidden">
                    <div
                      className={cn(
                        "h-full transition-all duration-500",
                        isPrimary ? "bg-danger" : pct > 20 ? "bg-warning" : "bg-ink-faint"
                      )}
                      style={{ width: `${pct}%` }}
                    />
                    {ciLo != null && ciHi != null && (
                      <div
                        className="absolute top-1/2 -translate-y-1/2 h-3 border-l border-r border-ink-dim opacity-50 pointer-events-none"
                        style={{
                          left: `${ciLo}%`,
                          width: `${Math.max(0, ciHi - ciLo)}%`,
                        }}
                      />
                    )}
                  </div>
                  {/* Plain-English subtitle so each bar reads on its own */}
                  <div className="text-2xs text-ink-faint mt-1 leading-snug max-w-md">
                    {CAUSE_PLAIN[key] ?? ""}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-4 pt-3 border-t border-hairline text-2xs text-ink-faint leading-relaxed">
            <span className="font-mono uppercase tracking-wider">Bayesian model:</span>{" "}
            discrete prior over 5 cause classes (informed by fairness-ML literature) ×
            independent Gaussian likelihoods over 6 standardized observables, posterior
            via exact Bayes' rule, 95% credible intervals from Monte Carlo over
            observable uncertainty (n=2,000 samples). Bars show posterior mean; whiskers
            show credible interval.
          </div>
        </Card>

        {/* Statistical validation — each metric annotated for non-ML readers */}
        <Card className="p-5 space-y-5">
          <div title={METRIC_HELP.flipRate}>
            <div className="flex items-baseline justify-between mb-2">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Counterfactual flip rate
              </div>
              <div className="text-2xs text-ink-faint">would change?</div>
            </div>
            <div className={cn(
              "font-mono tabular text-2xl",
              (firstAttr.counterfactual_flip_rate ?? 0) > 0.2 ? "text-danger" :
              (firstAttr.counterfactual_flip_rate ?? 0) > 0.1 ? "text-warning" : "text-success"
            )}>
              {firstAttr.counterfactual_flip_rate != null
                ? `${(firstAttr.counterfactual_flip_rate * 100).toFixed(1)}%`
                : "—"}
            </div>
            <div className="text-xs text-ink-muted mt-1 leading-snug">
              In plain English: if a person's most group-correlated features were
              swapped to look like the other group's, this is how often the model
              would change its decision.
            </div>
          </div>
          <div title={METRIC_HELP.permTest}>
            <div className="flex items-baseline justify-between mb-2">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Permutation test
              </div>
              <div className="text-2xs text-ink-faint">real or random?</div>
            </div>
            <div className={cn(
              "font-mono tabular text-2xl",
              firstAttr.permutation_test.significant ? "text-warning" : "text-success"
            )}>
              {firstAttr.permutation_test.p_value != null
                ? `p = ${firstAttr.permutation_test.p_value < 0.001 ? "<0.001" : firstAttr.permutation_test.p_value.toFixed(3)}`
                : "—"}
            </div>
            <div className="text-xs text-ink-muted mt-1 leading-snug">
              {firstAttr.permutation_test.significant
                ? "The pattern is statistically real, not random noise."
                : "The pattern could plausibly be random noise."}
              {" "}Based on{" "}
              <span className="font-mono">{firstAttr.permutation_test.n_permutations}</span>{" "}
              shuffled re-tests.
            </div>
          </div>
          <div title={METRIC_HELP.baseRateGap}>
            <div className="flex items-baseline justify-between mb-2">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Base-rate gap between groups
              </div>
              <div className="text-2xs text-ink-faint">in the data</div>
            </div>
            <div className="font-mono tabular text-2xl text-ink">
              {firstAttr.base_rate_gap != null
                ? `${(firstAttr.base_rate_gap * 100).toFixed(1)}pp`
                : "—"}
            </div>
            <div className="text-xs text-ink-muted mt-1 leading-snug">
              {firstAttr.groups.join(" vs ")} — measured in the data itself,
              before any model is trained.
            </div>
          </div>
        </Card>
      </div>

      {/* SHAP rank delta / correlated features */}
      {response.shap_available && firstAttr.proxy_features.length > 0 ? (
        <>
          <div className="rounded-md border border-hairline bg-elevated/30 px-4 py-3 text-xs text-ink-muted leading-relaxed">
            <span className="text-ink font-medium">What the next table shows: </span>
            features the model relies on much more for one group than the other.
            A big imbalance is a fingerprint of <em className="text-ink not-italic">proxy
            discrimination</em> — the feature is acting as a stand-in for the
            protected attribute. Ratio = how many times more important the feature
            is for one group vs. the other.
          </div>
          <ProxyFeaturesPanel
            proxyFeatures={firstAttr.proxy_features}
            groupShap={firstAttr.group_shap}
            groups={firstAttr.groups}
          />
        </>
      ) : firstAttr.correlated_features.length > 0 ? (
        <>
          <div className="rounded-md border border-hairline bg-elevated/30 px-4 py-3 text-xs text-ink-muted leading-relaxed">
            <span className="text-ink font-medium">What the next table shows: </span>
            features whose typical values differ most between the two groups in
            the data itself. SMD (standardized mean difference) of 0.5+ is large.
            Big gaps here mean the feature carries information about the
            protected group — even though that group label was excluded from training.
          </div>
          <CorrFeaturesPanel corrFeatures={firstAttr.correlated_features} />
        </>
      ) : null}
    </div>
  );
}

/** Group-context panel for Stage 5 — names each group with its human label,
 *  raw value, sample count, and positive rate. Anchors the forensic summary
 *  to concrete numbers a non-ML reader can verify. */
function GroupContextPanel({
  attrName,
  sizes,
  posRates,
}: {
  attrName: string;
  sizes: Record<string, number>;
  posRates: Record<string, number | null>;
}) {
  const entries = Object.entries(sizes).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return null;
  const total = entries.reduce((s, [, n]) => s + n, 0) || 1;
  const anyUnknown = entries.some(([raw]) => !formatGroup(attrName, raw).known);
  return (
    <div className="rounded-md border border-hairline overflow-hidden">
      <div className="px-4 py-2.5 bg-elevated/30 border-b border-hairline flex items-baseline justify-between">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Groups in {attrLabel(attrName)} ({attrName})
        </div>
        <div className="text-2xs font-mono text-ink-faint">
          encoded value · meaning · sample count
        </div>
      </div>
      <table className="w-full text-sm tabular">
        <thead>
          <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
            <th className="text-left font-normal py-2 pl-4 pr-3">Encoded</th>
            <th className="text-left font-normal py-2 px-3">Meaning</th>
            <th className="text-right font-normal py-2 px-3">n</th>
            <th className="text-right font-normal py-2 px-3">Share</th>
            <th className="text-right font-normal py-2 px-3 pr-4">P(+)</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([raw, n], i) => {
            const fmt = formatGroup(attrName, raw);
            const share = (n / total) * 100;
            const isMinority = i === entries.length - 1 && entries.length >= 2;
            const isMajority = i === 0 && entries.length >= 2;
            const pos = posRates[raw];
            return (
              <tr key={raw} className="border-b border-hairline last:border-0">
                <td className="py-2 pl-4 pr-3 font-mono text-xs">
                  {attrName} = {raw}
                </td>
                <td className="py-2 px-3 text-xs">
                  <span className="text-ink">{fmt.display ?? raw}</span>
                  {!fmt.known && (
                    <span className="ml-1.5 text-ink-faint text-2xs" title={UNKNOWN_MAPPING_HINT}>
                      ⓘ
                    </span>
                  )}
                </td>
                <td className="py-2 px-3 text-right font-mono text-xs">
                  {n.toLocaleString()}
                  {isMinority && (
                    <span className="ml-1.5 text-warning text-2xs uppercase tracking-wider">minority</span>
                  )}
                  {isMajority && (
                    <span className="ml-1.5 text-ink-faint text-2xs uppercase tracking-wider">majority</span>
                  )}
                </td>
                <td className="py-2 px-3 text-right font-mono text-xs text-ink-dim">
                  {share.toFixed(1)}%
                </td>
                <td className="py-2 px-3 pr-4 text-right font-mono text-xs">
                  {pos != null ? `${(pos * 100).toFixed(0)}%` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {anyUnknown && (
        <div className="px-4 py-2 bg-elevated/20 text-2xs text-ink-faint border-t border-hairline">
          Some encoded values don't have a known meaning in our label dictionary —
          add metadata for full interpretability.
        </div>
      )}
    </div>
  );
}

function ProxyFeaturesPanel({
  proxyFeatures,
  groupShap,
  groups,
}: {
  proxyFeatures: Stage5ProxyFeature[];
  groupShap: Record<string, { importance: Record<string, number | null>; ranks: Record<string, number> }>;
  groups: string[];
}) {
  const ga = groups[0];
  const gb = groups[1];
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Proxy discrimination · SHAP importance ratio ≥ 2×
        </div>
        <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
          mean |SHAP| per group
        </div>
      </div>
      <div className="overflow-x-auto rounded-md border border-hairline">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
              <th className="text-left font-normal py-2.5 pl-4 pr-3">Feature</th>
              <th className="text-right font-normal py-2.5 px-3">
                Imp · {ga}
              </th>
              <th className="text-right font-normal py-2.5 px-3">
                Rank · {ga}
              </th>
              {gb && <th className="text-right font-normal py-2.5 px-3">Imp · {gb}</th>}
              {gb && <th className="text-right font-normal py-2.5 px-3">Rank · {gb}</th>}
              <th className="text-right font-normal py-2.5 px-3 pr-4">Ratio</th>
            </tr>
          </thead>
          <tbody>
            {proxyFeatures.map((f) => {
              const impA = f.importances[ga];
              const impB = gb ? f.importances[gb] : null;
              const rkA = groupShap[ga]?.ranks[f.feature];
              const rkB = gb ? groupShap[gb]?.ranks[f.feature] : null;
              return (
                <tr key={f.feature} className="border-b border-hairline last:border-0">
                  <td className="py-2.5 pl-4 pr-3 font-mono text-xs text-ink">{f.feature}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-xs">
                    {impA != null ? impA.toFixed(4) : "—"}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-xs text-ink-dim">
                    {rkA != null ? `#${rkA}` : "—"}
                  </td>
                  {gb && (
                    <td className="py-2.5 px-3 text-right font-mono text-xs">
                      {impB != null ? impB.toFixed(4) : "—"}
                    </td>
                  )}
                  {gb && (
                    <td className="py-2.5 px-3 text-right font-mono text-xs text-ink-dim">
                      {rkB != null ? `#${rkB}` : "—"}
                    </td>
                  )}
                  <td className="py-2.5 px-3 pr-4 text-right font-mono text-xs">
                    <span className={cn(
                      (f.ratio ?? 0) >= 3 ? "text-danger" : "text-warning"
                    )}>
                      {f.ratio != null ? `${f.ratio.toFixed(1)}×` : "—"}
                    </span>
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

function CorrFeaturesPanel({ corrFeatures }: { corrFeatures: Stage5CorrFeature[] }) {
  if (corrFeatures.length === 0) return null;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Top features correlated with protected attribute
        </div>
        <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
          standardized mean difference
        </div>
      </div>
      <div className="overflow-x-auto rounded-md border border-hairline">
        <table className="w-full text-sm tabular">
          <thead>
            <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
              <th className="text-left font-normal py-2.5 pl-4 pr-3">Feature</th>
              <th className="text-right font-normal py-2.5 px-3">Mean · {corrFeatures[0]?.group_a}</th>
              <th className="text-right font-normal py-2.5 px-3">Mean · {corrFeatures[0]?.group_b}</th>
              <th className="text-right font-normal py-2.5 px-3 pr-4">SMD</th>
            </tr>
          </thead>
          <tbody>
            {corrFeatures.map((f) => (
              <tr key={f.feature} className="border-b border-hairline last:border-0">
                <td className="py-2.5 pl-4 pr-3 font-mono text-xs text-ink">{f.feature}</td>
                <td className="py-2.5 px-3 text-right font-mono text-xs">
                  {f.mean_a != null ? f.mean_a.toFixed(3) : "—"}
                </td>
                <td className="py-2.5 px-3 text-right font-mono text-xs">
                  {f.mean_b != null ? f.mean_b.toFixed(3) : "—"}
                </td>
                <td className="py-2.5 px-3 pr-4 text-right font-mono text-xs">
                  <span className={cn(
                    (f.smd ?? 0) >= 0.5 ? "text-danger" :
                    (f.smd ?? 0) >= 0.2 ? "text-warning" : "text-ink"
                  )}>
                    {f.smd != null ? f.smd.toFixed(3) : "—"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* Stage 6 — Guided remediation (real data from Flask /api/audit/stage/6) */

const CAUSE_LABEL_SHORT: Record<string, string> = {
  proxy_discrimination: "Proxy discrimination",
  representation_bias: "Representation bias",
  label_bias: "Label bias",
  threshold_effect: "Threshold effect",
  model_complexity_bias: "Model complexity bias",
};

/** Plain-English meaning of each action status, shown as a subtitle on every card. */
const STATUS_HELP: Record<Stage6ActionStatus, string> = {
  recommended: "Targets the actual root cause — apply this one",
  optional: "May help on the side, but isn't the main lever",
  blocked: "Would hide the problem rather than fix it",
};

const STATUS_LABEL: Record<Stage6ActionStatus, string> = {
  recommended: "Recommended",
  optional: "Optional",
  blocked: "Blocked",
};

function RemediationArtifact({ response }: { response: Stage6Response }) {
  const rec = response.actions.filter((a) => a.status === "recommended");
  const opt = response.actions.filter((a) => a.status === "optional");
  const blocked = response.actions.filter((a) => a.status === "blocked");

  return (
    <div className="space-y-5">
      {/* Plain-English summary — orients non-ML readers before the technical detail */}
      <div className="rounded-md border border-accent/20 bg-accent-soft/30 px-5 py-4">
        <div className="flex items-baseline gap-2 mb-1.5">
          <span className="text-2xs font-mono uppercase tracking-[0.18em] text-accent">
            In plain English
          </span>
          <span className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
            what should we do about it?
          </span>
        </div>
        <p className="text-sm text-ink leading-relaxed">
          Now that the audit knows <em className="not-italic text-ink">why</em>{" "}
          the model is biased, it suggests only the fixes that{" "}
          <span className="font-medium">
            actually address that specific cause
          </span>
          {" "}— and explicitly blocks the ones that would{" "}
          <span className="font-medium">just hide the problem</span> or create
          new legal/ethical risk.
          {response.safe_to_auto_fix ? (
            <span className="text-success">
              {" "}The recommended fix here is safe to apply automatically.
            </span>
          ) : (
            <span className="text-warning">
              {" "}A human should sign off before any of these are deployed.
            </span>
          )}
        </p>
      </div>

      {/* Summary header */}
      <div className="grid lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2 p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
              Diagnosis · {CAUSE_LABEL_SHORT[response.primary_root_cause] ?? response.primary_root_cause}
            </span>
            <div className="flex items-center gap-2">
              {response.safe_to_auto_fix ? (
                <span title="Confidence in the diagnosis is high enough that the recommended fix can run without human sign-off">
                  <Badge tone="success" dot>Safe to apply</Badge>
                </span>
              ) : (
                <span title="The cause type or confidence level means a person should review before applying any fix">
                  <Badge tone="warning">Manual review required</Badge>
                </span>
              )}
            </div>
          </div>
          <div className="font-serif text-xl text-ink mb-2">{response.diagnosis}</div>
          <p className="text-sm text-ink-muted leading-relaxed max-w-2xl">{response.summary}</p>
          <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-2xs font-mono text-ink-dim uppercase tracking-wider">
            <span>model: <span className="text-ink">{response.model_name}</span></span>
            <span>
              <span className="text-success">{rec.length} recommended</span>
              {" · "}
              <span className="text-ink-muted">{opt.length} optional</span>
              {" · "}
              <span className={cn(blocked.length > 0 ? "text-danger" : "text-ink-muted")}>
                {blocked.length} blocked
              </span>
            </span>
          </div>
        </Card>

        {/* Safety panel */}
        <Card className="p-5 space-y-4">
          <div>
            <div className="flex items-baseline justify-between mb-2">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Automated fix
              </div>
              <div className="text-2xs text-ink-faint">human needed?</div>
            </div>
            <div className={cn(
              "font-mono tabular text-2xl",
              response.safe_to_auto_fix ? "text-success" : "text-warning"
            )}>
              {response.safe_to_auto_fix ? "Approved" : "Blocked"}
            </div>
            <div className="text-xs text-ink-muted mt-1 leading-snug">
              {response.safe_to_auto_fix
                ? "The audit is confident enough (≥ 70%) that the recommended fix can be applied and verified automatically."
                : "Either the cause requires human judgment, or the audit isn't confident enough to apply a fix without oversight."}
            </div>
          </div>
          <div>
            <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
              Root cause class
            </div>
            <div className="font-mono text-sm text-ink">
              {response.primary_root_cause.replace(/_/g, " ")}
            </div>
            <div className="text-xs text-ink-faint mt-1">
              Carried over from Stage 5's diagnosis
            </div>
          </div>
          {response.warning && (
            <div className="rounded-md border border-warning/30 bg-warning/[0.06] px-3 py-2.5 text-xs text-warning leading-relaxed">
              <div className="font-medium uppercase tracking-wider text-2xs mb-1">
                Heads-up
              </div>
              {response.warning}
            </div>
          )}
        </Card>
      </div>

      {/* Action cards — sorted recommended → optional → blocked, with plain-language status */}
      <div className="grid lg:grid-cols-3 gap-4">
        {[...rec, ...opt, ...blocked].map((action) => (
          <RemediationCard key={action.id} action={action} />
        ))}
      </div>
    </div>
  );
}

function RemediationCard({ action }: { action: Stage6Action }) {
  // Recommended → green (success), Optional → yellow (warning),
  // Blocked → red (danger). Mirrors Stage 4's verdict colors so the visual
  // language stays consistent across stages.
  const tone =
    action.status === "blocked"
      ? "danger"
      : action.status === "recommended"
      ? "success"
      : "warning";
  const hasCriteria = action.success_criteria && action.success_criteria.length > 0;
  return (
    <Card className="p-5">
      <div className="flex items-center justify-between mb-3">
        <span title={STATUS_HELP[action.status]}>
          <Badge tone={tone} dot>{STATUS_LABEL[action.status]}</Badge>
        </span>
        <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
          Stage 6
        </span>
      </div>
      {/* Plain-English meaning of the badge so non-ML readers know what it implies */}
      <div className="text-2xs text-ink-faint uppercase tracking-wider mb-2">
        {STATUS_HELP[action.status]}
      </div>
      <div className="font-serif text-base text-ink mb-2">{action.title}</div>
      <p className="text-sm text-ink-muted leading-relaxed">{action.body}</p>
      {hasCriteria && (
        <div className="mt-3 pt-3 border-t border-hairline">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">
            Accept this fix only if
          </div>
          <ul className="space-y-1">
            {action.success_criteria.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-xs text-ink-muted leading-snug"
              >
                <span className="text-success font-mono mt-0.5">✓</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {action.status === "blocked" && !hasCriteria && (
        <div className="mt-3 pt-3 border-t border-hairline text-2xs text-ink-faint uppercase tracking-wider">
          No fix is being recommended — there are no acceptance criteria.
        </div>
      )}
    </Card>
  );
}

/* Stage 7 — Reasoning & validation checkpoints (real backend) */

function ReasoningCheckpointsArtifact({ response }: { response: Stage7Response }) {
  const cp1 = response.bias_validation;
  const cp2 = response.model_hypotheses;
  const cp3 = response.root_cause_consistency;
  const cp4 = response.final_recommendation;
  const nar = response.narratives;
  const usingGemini = nar.llm_provider === "gemini";

  const cp1Passed = cp1.severity !== "high" && cp1.inconsistencies.length === 0;
  const cp2Passed = cp2.verdict_summary === "confirmed" || cp2.verdict_summary === "rejected";
  const cp3Passed = !cp3.disagreement_flag;
  const cp4Passed = cp4.fairness_compliant && cp4.pareto_status === "non-dominated";

  return (
    <div className="space-y-5">
      {/* Plain-English summary panel — Gemini-generated when available */}
      <div className="rounded-md border border-accent/20 bg-accent-soft/30 px-5 py-4">
        <div className="flex items-baseline justify-between gap-2 mb-1.5">
          <div className="flex items-baseline gap-2">
            <span className="text-2xs font-mono uppercase tracking-[0.18em] text-accent">
              In plain English
            </span>
            <span className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
              does the audit hold up?
            </span>
          </div>
          <LlmProviderBadge provider={nar.llm_provider} model={nar.llm_model} />
        </div>
        <p className="text-sm text-ink leading-relaxed mb-2">
          {nar.executive_narrative}
        </p>
        <p className="text-xs text-ink-muted leading-relaxed">
          Verdicts on every checkpoint come from{" "}
          <span className="text-ink font-medium">Pydantic-validated deterministic logic</span>{" "}
          — the LLM only writes the prose, never the pass/fail. Without an API key the prose
          falls back to templates and the audit still works.{" "}
          {response.all_checkpoints_passed ? (
            <span className="text-success">All four checkpoints passed.</span>
          ) : (
            <span className="text-warning">
              {[cp1Passed, cp2Passed, cp3Passed, cp4Passed].filter(Boolean).length}/4 checkpoints passed
              — the failing ones are surfaced below.
            </span>
          )}
        </p>
        {!usingGemini && (
          <div className="mt-2 text-2xs text-ink-faint leading-snug">
            ⓘ To enable Gemini narratives, copy <span className="font-mono">backend/.env.example</span> to{" "}
            <span className="font-mono">backend/.env</span> and set{" "}
            <span className="font-mono">GEMINI_API_KEY</span>.
          </div>
        )}
      </div>

      <div className="grid sm:grid-cols-2 gap-3">
        <CheckpointCard
          n="1"
          title="Bias-fingerprint validation"
          subtitle="Recompute Stage 1 stats and flag internal contradictions"
          passed={cp1Passed}
          severity={cp1.severity}
          narrative={nar.cp1}
          usingGemini={usingGemini}
          body={cp1.summary}
          extra={
            cp1.inconsistencies.length > 0 ? (
              <ul className="space-y-1">
                {cp1.inconsistencies.map((i, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-2xs text-warning leading-snug">
                    <span className="font-mono mt-0.5">⚠</span>
                    <span>{i}</span>
                  </li>
                ))}
              </ul>
            ) : null
          }
        />
        <CheckpointCard
          n="2"
          title="Model-hypothesis consistency"
          subtitle="Does higher AUC come with worse fairness on this dataset?"
          passed={cp2Passed}
          narrative={nar.cp2}
          usingGemini={usingGemini}
          body={
            <>
              {cp2.hypothesis}
              {cp2.correlation_auc_eo != null && (
                <>
                  {" "}Pearson r ={" "}
                  <span className="font-mono text-ink">
                    {cp2.correlation_auc_eo.toFixed(3)}
                  </span>{" "}
                  · verdict:{" "}
                  <span
                    className={cn(
                      "font-mono",
                      cp2.verdict_summary === "confirmed" && "text-success",
                      cp2.verdict_summary === "rejected" && "text-warning",
                      cp2.verdict_summary === "ambiguous" && "text-ink-dim"
                    )}
                  >
                    {cp2.verdict_summary}
                  </span>
                </>
              )}
            </>
          }
          extra={
            cp2.per_model.length > 0 ? (
              <CP2PerModelGrid rows={cp2.per_model} />
            ) : null
          }
        />
        <CheckpointCard
          n="3"
          title="Root-cause cross-validation"
          subtitle="Statistical evidence vs SHAP-based diagnosis"
          passed={cp3Passed}
          narrative={nar.cp3}
          usingGemini={usingGemini}
          body={
            <>
              <div>
                Statistical:{" "}
                <span className="font-mono text-ink">
                  {cp3.statistical_root_cause.replace(/_/g, " ")}
                </span>
              </div>
              <div>
                ML-inferred:{" "}
                <span className="font-mono text-ink">
                  {cp3.ml_inferred_root_cause.replace(/_/g, " ")}
                </span>
              </div>
              <div className="mt-1">
                {cp3.agree ? (
                  <span className="text-success">Diagnoses agree</span>
                ) : (
                  <span className="text-warning">Disagreement — both reported</span>
                )}
              </div>
            </>
          }
          extra={
            cp3.notes.length > 0 || cp3.statistical_evidence.length > 0 ? (
              <ul className="space-y-1">
                {cp3.statistical_evidence.map((e, i) => (
                  <li key={`ev-${i}`} className="text-2xs text-ink-muted leading-snug">
                    · {e}
                  </li>
                ))}
                {cp3.notes.map((n, i) => (
                  <li key={`n-${i}`} className="text-2xs text-ink-muted leading-snug">
                    · {n}
                  </li>
                ))}
              </ul>
            ) : null
          }
        />
        <CheckpointCard
          n="4"
          title="Final recommendation gate"
          subtitle="Refuses dominated or fairness-failing recommendations"
          passed={cp4Passed}
          narrative={nar.cp4}
          usingGemini={usingGemini}
          body={
            <>
              {cp4.model ? (
                <div>
                  <span className="text-ink font-medium">{cp4.model}</span>
                  {" · AUC "}
                  <span className="font-mono">
                    {cp4.auc != null ? cp4.auc.toFixed(3) : "—"}
                  </span>
                  {" · EO gap "}
                  <span className="font-mono">
                    {cp4.eo_gap != null ? `${(cp4.eo_gap * 100).toFixed(1)}pp` : "—"}
                  </span>
                </div>
              ) : (
                <div className="text-warning">No model recommended</div>
              )}
              <div className="text-2xs text-ink-muted mt-1">{cp4.reason}</div>
            </>
          }
        />
      </div>
    </div>
  );
}

function CheckpointCard({
  n,
  title,
  subtitle,
  passed,
  severity,
  narrative,
  usingGemini,
  body,
  extra,
}: {
  n: string;
  title: string;
  subtitle: string;
  passed: boolean;
  severity?: "low" | "medium" | "high";
  narrative?: string;
  usingGemini?: boolean;
  body: React.ReactNode;
  extra?: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-md border bg-elevated/40 p-4 space-y-2",
        passed ? "border-success/20" : "border-warning/30"
      )}
    >
      <div className="flex items-start justify-between">
        <div className="text-2xs font-mono uppercase tracking-wider text-ink-dim">
          Checkpoint {n}
        </div>
        <div className="flex items-center gap-2">
          {severity && (
            <span
              className={cn(
                "text-2xs font-mono uppercase tracking-wider",
                severity === "high" && "text-danger",
                severity === "medium" && "text-warning",
                severity === "low" && "text-ink-dim"
              )}
            >
              severity: {severity}
            </span>
          )}
          {passed ? (
            <Badge tone="success" dot>Pass</Badge>
          ) : (
            <Badge tone="warning" dot>Review</Badge>
          )}
        </div>
      </div>
      <div className="text-sm font-medium text-ink">{title}</div>
      <div className="text-2xs text-ink-faint uppercase tracking-wider">
        {subtitle}
      </div>
      {narrative && (
        <div
          className={cn(
            "rounded-md px-3 py-2 text-xs leading-relaxed border-l-2",
            usingGemini
              ? "bg-accent-soft/30 border-accent text-ink"
              : "bg-elevated/60 border-hairline text-ink-muted"
          )}
        >
          {usingGemini && (
            <span className="text-2xs font-mono uppercase tracking-wider text-accent mr-1.5">
              Gemini ·
            </span>
          )}
          {narrative}
        </div>
      )}
      <div className="text-xs text-ink-muted leading-relaxed">{body}</div>
      {extra && <div className="pt-2 border-t border-hairline">{extra}</div>}
    </div>
  );
}

/** Small badge showing whether narratives in this stage came from Gemini or
 *  fell back to deterministic templates. Lets judges/users know at a glance. */
function LlmProviderBadge({
  provider,
  model,
}: {
  provider: "gemini" | "deterministic";
  model: string | null;
}) {
  if (provider === "gemini") {
    return (
      <span
        title={`Narratives generated by ${model ?? "Gemini"}; numbers and verdicts are deterministic.`}
      >
        <Badge tone="accent" dot>
          Gemini {model ?? ""}
        </Badge>
      </span>
    );
  }
  return (
    <span title="No GEMINI_API_KEY configured — narratives are template-based.">
      <Badge tone="neutral">Template prose</Badge>
    </span>
  );
}

function CP2PerModelGrid({ rows }: { rows: CP2PerModel[] }) {
  return (
    <div className="grid grid-cols-2 gap-1.5">
      {rows.map((r) => (
        <div
          key={r.model_key}
          className="flex items-baseline justify-between gap-1.5 text-2xs font-mono"
        >
          <span className="text-ink-muted truncate">{r.model_name}</span>
          <span
            className={cn(
              r.verdict === "confirmed" && "text-success",
              r.verdict === "rejected" && "text-warning",
              r.verdict === "ambiguous" && "text-ink-dim",
              r.verdict === "insufficient_data" && "text-ink-faint"
            )}
          >
            {r.verdict}
          </span>
        </div>
      ))}
    </div>
  );
}

/* Stage 8 — Decision-intelligence report (5 tabs, real backend) */

function ExecutiveReportArtifact({ response }: { response: Stage8Response }) {
  const [tab, setTab] = useState<"exec" | "fairness" | "behavior" | "actions" | "deploy">("exec");
  const TABS: { id: typeof tab; label: string; sub: string }[] = [
    { id: "exec",     label: "Executive",    sub: "Recommendation" },
    { id: "fairness", label: "Fairness & risk", sub: "Disparities" },
    { id: "behavior", label: "Model behavior",  sub: "What it relies on" },
    { id: "actions",  label: "Actions",         sub: "What to do" },
    { id: "deploy",   label: "Deployment",      sub: "Go / no-go" },
  ];
  const usingGemini = response.llm_provider === "gemini";

  return (
    <div className="space-y-5">
      <div className="rounded-md border border-accent/20 bg-accent-soft/30 px-5 py-4">
        <div className="flex items-baseline justify-between gap-2 mb-1.5">
          <div className="flex items-baseline gap-2">
            <span className="text-2xs font-mono uppercase tracking-[0.18em] text-accent">
              In plain English
            </span>
            <span className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
              decision-ready report
            </span>
          </div>
          <LlmProviderBadge provider={response.llm_provider} model={response.llm_model} />
        </div>
        {response.executive_narrative ? (
          <p className="text-sm text-ink leading-relaxed mb-2">
            <span className="text-2xs font-mono uppercase tracking-wider text-accent mr-1.5">
              Gemini ·
            </span>
            {response.executive_narrative}
          </p>
        ) : null}
        <p className="text-sm text-ink-muted leading-relaxed">
          Five tabs distilled from the eight statistical stages — written for a product manager,
          executive, or compliance reviewer. Every <span className="text-ink">number</span> is
          traceable back to a specific earlier stage;{" "}
          {usingGemini ? (
            <>the <span className="text-ink">prose</span> is written by Gemini under strict instructions to never invent or override the verdicts.</>
          ) : (
            <>the <span className="text-ink">prose</span> is template-based until you set <span className="font-mono">GEMINI_API_KEY</span> in <span className="font-mono">backend/.env</span>.</>
          )}
        </p>
      </div>

      {/* Tab nav */}
      <div className="flex flex-wrap gap-1.5 border-b border-hairline pb-0">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "px-3 py-2 text-xs font-medium rounded-t-md border-b-2 transition-colors",
              tab === t.id
                ? "border-accent text-ink bg-elevated/60"
                : "border-transparent text-ink-muted hover:text-ink hover:bg-elevated/30"
            )}
          >
            <div>{t.label}</div>
            <div className="text-2xs text-ink-faint font-mono uppercase tracking-wider">
              {t.sub}
            </div>
          </button>
        ))}
      </div>

      {tab === "exec" && <ExecutiveTab data={response.executive} />}
      {tab === "fairness" && <FairnessRiskTab data={response.fairness_risk} />}
      {tab === "behavior" && <ModelBehaviorTab data={response.model_behavior} />}
      {tab === "actions" && <ActionsTab data={response.actions} />}
      {tab === "deploy" && <DeploymentTab data={response.deployment} />}
    </div>
  );
}

function ExecutiveTab({ data }: { data: Stage8Response["executive"] }) {
  return (
    <Card className="p-5 space-y-4">
      <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
        Recommended model
      </div>
      <div className="font-serif text-2xl text-ink">
        {data.model ?? "No recommendation issued"}
      </div>
      {data.model && (
        <div className="grid sm:grid-cols-3 gap-4">
          <div>
            <div className="text-2xs text-ink-faint uppercase tracking-wider">AUC</div>
            <div className="font-mono tabular text-xl text-ink">
              {data.auc != null ? data.auc.toFixed(3) : "—"}
            </div>
          </div>
          <div>
            <div className="text-2xs text-ink-faint uppercase tracking-wider">EO gap</div>
            <div className="font-mono tabular text-xl text-ink">
              {data.eo_gap != null ? `${(data.eo_gap * 100).toFixed(1)}pp` : "—"}
            </div>
          </div>
          <div>
            <div className="text-2xs text-ink-faint uppercase tracking-wider">Status</div>
            <div className="font-mono text-xl text-ink">
              {data.status.replace(/_/g, " ")}
            </div>
          </div>
        </div>
      )}
      <div className="text-sm text-ink-muted leading-relaxed">{data.reason}</div>
      <div className="rounded-md border border-hairline bg-elevated/30 px-4 py-3">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">
          Business interpretation
        </div>
        <p className="text-sm text-ink leading-relaxed">
          {data.business_interpretation}
        </p>
      </div>
    </Card>
  );
}

function FairnessRiskTab({ data }: { data: Stage8Response["fairness_risk"] }) {
  const sevColor =
    data.severity === "high" ? "text-danger" :
    data.severity === "medium" ? "text-warning" : "text-success";
  return (
    <div className="space-y-4">
      <Card className="p-5 space-y-3">
        <div className="flex items-baseline justify-between">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            Risk severity
          </div>
          <div className={cn("font-mono uppercase tracking-wider text-sm", sevColor)}>
            {data.severity}
          </div>
        </div>
        <div className="font-serif text-lg text-ink">
          Primary bias type: {(data.primary_bias_type || "unknown").replace(/_/g, " ")}
        </div>
        {!data.diagnoses_agree && (
          <div className="rounded-md border border-warning/30 bg-warning/[0.06] px-3 py-2 text-xs text-warning">
            ⚠ Statistical and ML diagnoses disagree. Statistical:{" "}
            <span className="font-mono">{data.statistical_root_cause}</span>; ML-inferred:{" "}
            <span className="font-mono">{data.ml_inferred_root_cause}</span>. Manual review recommended.
          </div>
        )}
      </Card>

      {data.disadvantaged_groups.length > 0 && (
        <Card className="p-5">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
            Disadvantaged groups
          </div>
          <div className="space-y-2">
            {data.disadvantaged_groups.map((g) => {
              const minFmt = formatGroup(g.attribute, g.minority_group);
              const majFmt = formatGroup(g.attribute, g.majority_group);
              return (
                <div
                  key={`${g.attribute}-${g.minority_group}`}
                  className="text-sm text-ink-muted leading-relaxed"
                >
                  <span className="text-ink font-medium">
                    {minFmt.display ?? g.minority_group}
                  </span>{" "}
                  <span className="font-mono text-2xs text-ink-faint">
                    ({g.attribute} = {g.minority_group}, n = {g.minority_n})
                  </span>{" "}
                  is underrepresented vs.{" "}
                  <span className="text-ink">{majFmt.display ?? g.majority_group}</span>{" "}
                  <span className="font-mono text-2xs text-ink-faint">
                    ({g.attribute} = {g.majority_group}, n = {g.majority_n})
                  </span>
                  {g.imbalance_ratio != null && (
                    <span className="text-warning font-mono">
                      {" — "}
                      {g.imbalance_ratio.toFixed(1)}× ratio
                    </span>
                  )}
                  .
                </div>
              );
            })}
          </div>
        </Card>
      )}

      <Card className="p-5">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
          Statistical evidence
        </div>
        <div className="grid sm:grid-cols-2 gap-3 text-xs font-mono">
          <EvidenceRow
            label="χ² p-value"
            value={data.evidence.chi2_p_value != null ? data.evidence.chi2_p_value.toFixed(3) : "—"}
            highlight={data.evidence.chi2_significant ? "warning" : null}
          />
          <EvidenceRow
            label="Base-rate gap"
            value={data.evidence.base_rate_gap_pp != null ? `${data.evidence.base_rate_gap_pp.toFixed(1)}pp` : "—"}
          />
          <EvidenceRow
            label="Missingness disparity"
            value={data.evidence.missingness_disparity_pp != null ? `${data.evidence.missingness_disparity_pp.toFixed(1)}pp` : "—"}
          />
          <EvidenceRow
            label="Largest imbalance ratio"
            value={data.evidence.largest_imbalance_ratio != null ? `${data.evidence.largest_imbalance_ratio.toFixed(1)}×` : "—"}
          />
        </div>
        {data.inconsistencies.length > 0 && (
          <ul className="mt-3 pt-3 border-t border-hairline space-y-1">
            {data.inconsistencies.map((i, idx) => (
              <li key={idx} className="flex items-start gap-2 text-2xs text-warning leading-snug">
                <span className="font-mono mt-0.5">⚠</span>
                <span>{i}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function EvidenceRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: "warning" | null;
}) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-ink-dim uppercase tracking-wider text-2xs">{label}</span>
      <span className={cn("text-ink", highlight === "warning" && "text-warning")}>
        {value}
      </span>
    </div>
  );
}

function ModelBehaviorTab({ data }: { data: Stage8Response["model_behavior"] }) {
  return (
    <div className="space-y-4">
      <Card className="p-5">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
          What the model relies on
        </div>
        <p className="text-sm text-ink leading-relaxed">{data.narrative}</p>
        {!data.shap_available && (
          <div className="mt-3 text-2xs text-ink-faint uppercase tracking-wider">
            Note: SHAP wasn't available — feature importance is correlation-based.
          </div>
        )}
      </Card>

      {data.top_features.length > 0 && (
        <Card className="p-5">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
            Top features (across groups)
          </div>
          <div className="flex flex-wrap gap-1.5">
            {data.top_features.map((f) => (
              <span
                key={f}
                className="rounded-full border border-hairline bg-elevated/40 px-3 py-1 text-xs font-mono text-ink"
              >
                {f}
              </span>
            ))}
          </div>
        </Card>
      )}

      {data.proxy_features.length > 0 && (
        <Card className="p-5">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
            Possible proxy features
          </div>
          <p className="text-xs text-ink-muted mb-3 leading-relaxed">
            These features are notably more influential for one group than the
            other — even though the protected attribute itself was excluded from training.
            They may be acting as indirect signals.
          </p>
          <div className="space-y-2">
            {data.proxy_features.map((p) => (
              <div key={p.feature} className="flex items-baseline justify-between text-sm">
                <span className="font-mono text-ink">{p.feature}</span>
                <span className="font-mono text-warning">
                  {p.ratio != null ? `${p.ratio.toFixed(1)}×` : "—"}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function ActionsTab({ data }: { data: Stage8Response["actions"] }) {
  const isDataMode = data.mode === "data_remediation";
  return (
    <div className="space-y-4">
      {data.diagnosis && (
        <Card className="p-5 space-y-2">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            {isDataMode ? "Dataset diagnosis" : "Diagnosis"}
          </div>
          <div className="font-serif text-lg text-ink">{data.diagnosis}</div>
          {data.headline && (
            <div className="text-sm text-danger font-medium">{data.headline}.</div>
          )}
          {data.summary && (
            <p className="text-sm text-ink-muted leading-relaxed">{data.summary}</p>
          )}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs font-mono uppercase tracking-wider pt-1">
            <span className="text-success">{data.recommended_count} recommended</span>
            <span className="text-danger">{data.blocked_count} blocked</span>
            <span className={cn(data.safe_to_auto_fix ? "text-success" : "text-warning")}>
              {data.safe_to_auto_fix ? "safe to apply" : "manual review required"}
            </span>
            {data.verified_by_stage7 && (
              <span className="text-accent">verified by Stage 7</span>
            )}
          </div>
          {data.warning && (
            <div className="rounded-md border border-warning/30 bg-warning/[0.06] px-3 py-2 text-xs text-warning leading-relaxed">
              {data.warning}
            </div>
          )}
        </Card>
      )}

      <div className="grid lg:grid-cols-3 gap-4">
        {data.actions.map((a) =>
          isDataLevelAction(a) ? (
            <DataActionCard key={a.id} action={a} />
          ) : (
            <RemediationCard key={a.id} action={a} />
          )
        )}
      </div>
    </div>
  );
}

/** Type guard separating Stage 6 model-level actions (have `status`) from
 *  Stage 7 data-level actions (have `why_model_fix_insufficient`). */
function isDataLevelAction(
  a: Stage6Action | DataRemediationAction
): a is DataRemediationAction {
  return (a as DataRemediationAction).why_model_fix_insufficient !== undefined;
}

function DataActionCard({ action }: { action: DataRemediationAction }) {
  const hasCriteria = action.success_criteria && action.success_criteria.length > 0;
  return (
    <Card className="p-5 border-warning/25">
      <div className="flex items-center justify-between mb-3">
        <Badge tone="accent" dot>Apply at data layer</Badge>
        <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
          Stage 7
        </span>
      </div>
      <div className="font-serif text-base text-ink mb-2">{action.title}</div>
      <p className="text-sm text-ink-muted leading-relaxed">{action.body}</p>
      <div className="mt-3 pt-3 border-t border-hairline">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">
          Why a model-level fix isn&apos;t enough
        </div>
        <p className="text-xs text-ink-muted leading-snug">
          {action.why_model_fix_insufficient}
        </p>
      </div>
      {hasCriteria && (
        <div className="mt-3 pt-3 border-t border-hairline">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">
            Accept this fix only if
          </div>
          <ul className="space-y-1">
            {action.success_criteria.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-xs text-ink-muted leading-snug"
              >
                <span className="text-success font-mono mt-0.5">✓</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

function DeploymentTab({ data }: { data: Stage8Response["deployment"] }) {
  const verdictColor =
    data.verdict === "deploy" ? "text-success" :
    data.verdict === "conditional" ? "text-warning" : "text-danger";
  const VerdictBadge = (
    <Badge
      tone={
        data.verdict === "deploy" ? "success" :
        data.verdict === "conditional" ? "warning" : "danger"
      }
      dot
    >
      {data.verdict.replace(/_/g, " ")}
    </Badge>
  );
  return (
    <div className="space-y-4">
      <Card className="p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
            Deployment verdict
          </div>
          {VerdictBadge}
        </div>
        <div className={cn("font-serif text-2xl mb-2", verdictColor)}>
          {data.verdict.replace(/_/g, " ")}
        </div>
        <p className="text-sm text-ink-muted leading-relaxed mb-3">{data.verdict_text}</p>
        <div className="text-2xs font-mono text-ink-dim uppercase tracking-wider">
          {data.passed_count} of {data.total_conditions} conditions met
        </div>
      </Card>

      <Card className="p-5">
        <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-3">
          Deployment conditions
        </div>
        <div className="space-y-2">
          {data.conditions.map((c, i) => (
            <DeploymentConditionRow key={i} condition={c} />
          ))}
        </div>
      </Card>
    </div>
  );
}

function DeploymentConditionRow({ condition }: { condition: Stage8DeploymentCondition }) {
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2.5",
        condition.passed
          ? "border-success/20 bg-success/[0.04]"
          : "border-warning/30 bg-warning/[0.04]"
      )}
    >
      <div className="flex items-baseline justify-between">
        <div className="text-sm text-ink font-medium">{condition.name}</div>
        <span className={cn("text-2xs font-mono uppercase tracking-wider", condition.passed ? "text-success" : "text-warning")}>
          {condition.passed ? "✓ pass" : "⚠ fail"}
        </span>
      </div>
      {condition.detail && (
        <div className="text-2xs text-ink-muted mt-1">{condition.detail}</div>
      )}
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
