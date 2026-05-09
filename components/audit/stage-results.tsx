"use client";

import { ParetoChart } from "@/components/pareto-chart";
import { Leaderboard } from "@/components/leaderboard";
import { ReportTabs } from "@/components/report-tabs";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  Stat,
} from "@/components/primitives";
import {
  MODELS,
  POPULATION_IMPACT,
  REMEDIATION,
  ROOT_CAUSE,
} from "@/lib/data";
import type { ParsedCsv } from "@/lib/csv";
import type { AuditConfig } from "./stage-configure";

export function ResultsStage({
  csv,
  cfg,
  elapsedSec,
  onRestart,
}: {
  csv: ParsedCsv;
  cfg: AuditConfig;
  elapsedSec: string;
  onRestart: () => void;
}) {
  const recommended = MODELS.find((m) => m.recommended)!;
  const worst = [...MODELS].sort((a, b) => b.eqOddsGap - a.eqOddsGap)[0];

  return (
    <div className="animate-fade-up">
      {/* Title row */}
      <section className="border-b border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-8 sm:py-12">
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div className="space-y-3 max-w-2xl">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="success" dot>
                  Pareto-validated
                </Badge>
                <Badge tone="neutral">5 models · 7 metrics · {csv.headers.length} features</Badge>
                <Badge tone="neutral">FDR q = 0.05</Badge>
              </div>
              <h2 className="font-serif text-3xl text-ink leading-[1.05] tracking-tight">
                {csv.fileName.replace(/\.csv$/i, "")}
              </h2>
              <p className="text-sm text-ink-muted max-w-xl">
                {csv.rowCount.toLocaleString()} records · {csv.headers.length} features ·
                target <span className="font-mono text-ink">{cfg.target}</span> ·
                protected{" "}
                <span className="font-mono text-ink">
                  {cfg.protectedAttrs.join(", ")}
                </span>
                .
                Audit completed in {elapsedSec}s.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="secondary" size="md" onClick={onRestart}>
                Run another
              </Button>
              <Button variant="primary" size="md">
                Deploy recommended
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Hero summary row */}
      <section className="border-b border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-8 grid lg:grid-cols-12 gap-5">
          {/* Recommended model */}
          <Card className="lg:col-span-4 p-6 shadow-glow border-accent/30">
            <div className="flex items-center justify-between mb-5">
              <Badge tone="accent" dot>
                Recommended
              </Badge>
              <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
                Pareto-optimal
              </span>
            </div>
            <div className="font-serif text-2xl leading-tight tracking-tight mb-1">
              {recommended.name}
            </div>
            <div className="text-xs text-ink-muted mb-6">{recommended.family}</div>

            <div className="grid grid-cols-2 gap-x-6 gap-y-5 pt-5 border-t border-hairline">
              <Stat
                label="AUC-ROC"
                value={recommended.auc.toFixed(3)}
                ci={`${recommended.aucCi[0].toFixed(3)} – ${recommended.aucCi[1].toFixed(3)}`}
              />
              <Stat
                label="Eq. odds gap"
                value={recommended.eqOddsGap.toFixed(3)}
                ci={`${recommended.eqOddsCi[0].toFixed(3)} – ${recommended.eqOddsCi[1].toFixed(3)}`}
              />
              <Stat
                label="Calibration ECE"
                value={recommended.ece.toFixed(3)}
                hint="Hosmer-Lemeshow"
              />
              <Stat
                label="Disparate impact"
                value="0.917"
                hint="≥ 0.80 (4/5ths rule)"
              />
            </div>
          </Card>

          {/* Population impact */}
          <Card className="lg:col-span-5 p-6 grid-bg">
            <div className="flex items-center justify-between mb-5">
              <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Population impact
              </span>
              <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
                vs. worst alternative
              </span>
            </div>
            <div className="flex items-baseline gap-3 mb-4 flex-wrap">
              <div className="font-serif text-display text-ink leading-none italic">
                {POPULATION_IMPACT.toLocaleString()}
              </div>
              <div className="text-sm text-ink-muted max-w-[14rem]">
                people in this dataset receive a&nbsp;fairer decision under the recommended
                model than under <span className="text-ink">{worst.name}</span>.
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-4 mt-6 pt-5 border-t border-hairline text-xs">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                <span className="text-ink-muted">Recommended FPR gap</span>
                <span className="font-mono text-ink">0.041</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-danger" />
                <span className="text-ink-muted">Worst FPR gap</span>
                <span className="font-mono text-ink">0.118</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-ink-muted">Δ</span>
                <span className="font-mono text-success">−65.3%</span>
              </div>
            </div>
          </Card>

          {/* Root cause */}
          <Card className="lg:col-span-3 p-6">
            <div className="flex items-center justify-between mb-5">
              <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Root cause
              </span>
              <span className="text-2xs text-ink-dim font-mono uppercase tracking-wider">
                Bayesian
              </span>
            </div>
            <div className="font-serif text-xl text-ink mb-1">Proxy discrimination</div>
            <div className="text-xs text-ink-muted mb-6">
              A feature in your data correlates with{" "}
              <span className="font-mono text-ink">{cfg.protectedAttrs[0]}</span> and drives
              prediction differently across groups.
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
        </div>
      </section>

      {/* Pareto chart */}
      <section className="border-b border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-10">
          <Card>
            <CardHeader
              eyebrow="Stage 4 · Pareto frontier"
              title={
                <span>Accuracy × fairness, with bootstrap confidence ellipses</span>
              }
              meta="DIETTERICH 5×2 CV · n=2,000"
            />
            <div className="p-3 sm:p-6">
              <ParetoChart />
            </div>
            <div className="border-t border-hairline px-5 sm:px-6 py-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
              <Legend color="#5B7FFF" label="Recommended" filled />
              <Legend color="#34D399" label="Pareto-optimal" />
              <Legend color="#F87171" label="Higher accuracy, less fair" />
              <Legend color="#9CA3AF" label="Pareto-dominated" dim />
              <span className="ml-auto text-ink-dim font-mono text-2xs uppercase tracking-wider">
                Hover any model for full statistical detail
              </span>
            </div>
          </Card>
        </div>
      </section>

      {/* Leaderboard */}
      <section className="border-b border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-10">
          <Card>
            <CardHeader
              eyebrow="Stage 3 · Per-model fairness audit"
              title="Model leaderboard — every metric, every model"
              meta="BH FDR q=0.05 · CIs n=2,000"
            />
            <Leaderboard />
          </Card>
        </div>
      </section>

      {/* Remediation */}
      <section className="border-b border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-10">
          <div className="grid lg:grid-cols-3 gap-5">
            {REMEDIATION.map((r) => (
              <Card key={r.title} className="p-6">
                <div className="flex items-center justify-between mb-4">
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
                <div className="font-serif text-xl text-ink mb-2">{r.title}</div>
                <p className="text-sm text-ink-muted leading-relaxed">{r.body}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Three-audience reports */}
      <section>
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-10">
          <Card>
            <CardHeader
              eyebrow="Stage 8 · Three-audience report"
              title="Same statistics. Three communication registers."
              meta="GEMINI 1.5 PRO · Pydantic-validated"
            />
            <ReportTabs />
          </Card>
        </div>
      </section>

      <footer className="border-t border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-8 flex flex-wrap items-center justify-between gap-4 text-xs text-ink-dim">
          <div className="flex items-center gap-3 font-mono uppercase tracking-wider">
            <span>Built on</span>
            {["fairlearn", "scipy", "shap", "mlxtend", "Gemini 1.5", "MongoDB"].map((t) => (
              <span key={t} className="text-ink-muted">
                {t}
              </span>
            ))}
          </div>
          <span>The fence at the top of the cliff, not the ambulance at the bottom.</span>
        </div>
      </footer>
    </div>
  );
}

function Legend({
  color,
  label,
  filled,
  dim,
}: {
  color: string;
  label: string;
  filled?: boolean;
  dim?: boolean;
}) {
  return (
    <div className={`flex items-center gap-2 ${dim ? "opacity-60" : ""}`}>
      <span
        className="w-2.5 h-2.5 rounded-full"
        style={{
          background: filled ? color : "transparent",
          border: `1.5px solid ${color}`,
        }}
      />
      <span className="text-ink-muted">{label}</span>
    </div>
  );
}
