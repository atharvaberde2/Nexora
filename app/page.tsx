import Link from "next/link";
import { Nav } from "@/components/nav";
import { Badge, Button, Card } from "@/components/primitives";

const STAGES = [
  { n: "01", title: "Bias fingerprinting", body: "Audit what the data already encodes — class imbalance, missingness disparity, label-bias chi-square, per-subgroup power analysis." },
  { n: "02", title: "Multi-model training", body: "Five model families trained simultaneously under identical conditions. Same seeds, same folds, same data order. Apples to apples." },
  { n: "03", title: "Per-model fairness audit", body: "Bootstrap CIs (n=2,000), DeLong, McNemar, BH FDR correction across every model × metric × group combination." },
  { n: "04", title: "Pareto frontier", body: "Dietterich 5×2 CV paired t-test applied to fairness metrics — the statistically correct comparison no other tool generates." },
  { n: "05", title: "Root cause diagnosis", body: "5-class Bayesian classifier outputs a probability distribution over the five identifiable failure modes." },
  { n: "06", title: "Guided remediation", body: "Root-cause-conditional fixes. The system knows when not to fix — which is as important as knowing what to fix." },
  { n: "07", title: "Gemini reasoning layer", body: "Four schema-validated checkpoints. Pydantic enforcement. Mathematically constrained to the Pareto-optimal set." },
  { n: "08", title: "Three-audience report", body: "Engineer scorecard, regulator submission, community plain-language report. Same statistics, three registers." },
];

const COMPETITORS = [
  { name: "Nexora", trains: true, ci: true, dietterich: true, pareto: true, fdr: true, root: true, remed: true, prevention: true },
  { name: "Fairlearn", trains: false, ci: false, dietterich: false, pareto: false, fdr: false, root: false, remed: "partial", prevention: false },
  { name: "AuditML", trains: false, ci: true, dietterich: false, pareto: false, fdr: true, root: true, remed: false, prevention: false },
  { name: "PRISM", trains: false, ci: false, dietterich: false, pareto: false, fdr: false, root: false, remed: false, prevention: true },
  { name: "Aequitas", trains: false, ci: false, dietterich: false, pareto: false, fdr: false, root: false, remed: false, prevention: false },
] as const;

export default function Landing() {
  return (
    <main className="min-h-screen">
      <Nav />

      {/* Hero */}
      <section className="relative">
        <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-accent/10 blur-[120px] rounded-full pointer-events-none" />

        <div className="relative max-w-[1320px] mx-auto px-5 sm:px-8 pt-20 sm:pt-28 pb-20 sm:pb-28">
          <div className="max-w-3xl">
            <Badge tone="neutral" className="mb-8">
              <span className="text-ink-dim">Demo · </span>
              <span className="ml-1 text-ink">HackDavis 2026</span>
            </Badge>

            <h1 className="font-serif text-[44px] sm:text-[64px] lg:text-[88px] leading-[0.98] tracking-[-0.03em] text-ink mb-6">
              Five models enter.
              <br />
              <span className="italic text-ink-muted">The fair one</span>{" "}
              <span className="italic">ships.</span>
            </h1>

            <p className="text-md sm:text-lg text-ink-muted max-w-2xl leading-relaxed mb-10">
              Nexora trains five model families on your data and proves —
              statistically — which one is fair to deploy. Before a single real
              person is affected.
            </p>

            <div className="flex flex-wrap items-center gap-3">
              <Link href="/audit">
                <Button variant="primary" size="md">
                  Start a fairness audit
                  <ArrowRight />
                </Button>
              </Link>
              <span className="text-xs text-ink-dim ml-1 hidden sm:inline-flex items-center gap-2">
                <span className="w-1 h-1 rounded-full bg-ink-faint" />
                Bring your CSV — or use the COMPAS sample
              </span>
            </div>
          </div>

          {/* Inline value strip */}
          <div className="mt-20 sm:mt-24 grid grid-cols-2 lg:grid-cols-4 gap-px bg-hairline border border-hairline rounded-lg overflow-hidden">
            {[
              { kpi: "5", unit: "model families", note: "trained simultaneously, identical conditions" },
              { kpi: "2,000", unit: "bootstrap iters", note: "on every gap, every group, every model" },
              { kpi: "p < 0.05", unit: "Dietterich 5×2 CV", note: "applied to fairness metrics — first of its kind" },
              { kpi: "EN + ES", unit: "audio reports", note: "plain-language, ElevenLabs-narrated" },
            ].map((s) => (
              <div key={s.kpi} className="bg-canvas px-5 py-6">
                <div className="font-serif text-2xl text-ink mb-1">{s.kpi}</div>
                <div className="text-xs text-ink-muted uppercase tracking-wider mb-2">{s.unit}</div>
                <div className="text-xs text-ink-dim leading-relaxed">{s.note}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The problem */}
      <section className="border-t border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-20">
          <div className="grid lg:grid-cols-12 gap-10 items-start">
            <div className="lg:col-span-5">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-4">The gap</div>
              <h2 className="font-serif text-2xl leading-tight tracking-tight mb-6">
                Every harmful deployment starts with a model selection decision nobody audits.
              </h2>
              <p className="text-md text-ink-muted leading-relaxed">
                PRISM audits your dataset before training. Fairlearn audits a
                single model after it's chosen. Nobody answers the question in
                the middle: <span className="text-ink">of all the models you could deploy, which one is fair to ship?</span>
              </p>
            </div>

            <div className="lg:col-span-7 space-y-3">
              {[
                { stage: "BEFORE", tool: "PRISM", detail: "Audits your dataset before training — before any model exists.", tone: "neutral" as const },
                { stage: "MIDDLE", tool: "Nexora", detail: "Compares five trained models, statistically, before deployment.", tone: "accent" as const, recommended: true },
                { stage: "AFTER", tool: "Fairlearn · AuditML · Aequitas", detail: "Audit a single deployed model. After harm has occurred.", tone: "neutral" as const },
              ].map((row) => (
                <div
                  key={row.stage}
                  className={`grid grid-cols-[88px_1fr_auto] gap-4 items-center px-5 py-4 rounded-md border ${
                    row.recommended
                      ? "border-accent/40 bg-accent-soft"
                      : "border-hairline bg-surface/50"
                  }`}
                >
                  <span className="font-mono text-2xs uppercase tracking-[0.18em] text-ink-dim">
                    {row.stage}
                  </span>
                  <div>
                    <div className={`text-sm font-medium ${row.recommended ? "text-ink" : "text-ink"}`}>
                      {row.tool}
                    </div>
                    <div className="text-xs text-ink-muted mt-0.5">{row.detail}</div>
                  </div>
                  {row.recommended && (
                    <Badge tone="accent" dot>Nexora</Badge>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 8-stage pipeline */}
      <section className="border-t border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-20">
          <div className="max-w-2xl mb-12">
            <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-4">Pipeline</div>
            <h2 className="font-serif text-2xl leading-tight tracking-tight">
              Upload one CSV. Eight stages. One recommendation that survives statistical scrutiny.
            </h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-px bg-hairline border border-hairline rounded-lg overflow-hidden">
            {STAGES.map((s) => (
              <div key={s.n} className="bg-canvas p-6 group hover:bg-surface transition-colors">
                <div className="flex items-center justify-between mb-5">
                  <span className="font-mono text-2xs text-ink-dim tracking-wider">STAGE {s.n}</span>
                  <span className="w-1 h-1 rounded-full bg-ink-faint group-hover:bg-accent transition-colors" />
                </div>
                <h3 className="font-medium text-ink mb-2">{s.title}</h3>
                <p className="text-sm text-ink-muted leading-relaxed">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Differentiator table */}
      <section className="border-t border-hairline bg-surface/40">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-20">
          <div className="max-w-2xl mb-10">
            <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-4">Why Nexora</div>
            <h2 className="font-serif text-2xl leading-tight tracking-tight">
              The statistical engine no fairness tool ships.
            </h2>
          </div>

          <Card>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
                    <th className="text-left font-normal py-4 pl-5 pr-3">Capability</th>
                    {COMPETITORS.map((c) => (
                      <th key={c.name} className={`font-normal py-4 px-3 text-center ${c.name === "Nexora" ? "text-ink" : ""}`}>
                        {c.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { label: "Trains + audits multiple models", key: "trains" },
                    { label: "Bootstrap CIs on all gaps", key: "ci" },
                    { label: "Dietterich model comparison", key: "dietterich" },
                    { label: "Pareto frontier with bootstrap", key: "pareto" },
                    { label: "BH FDR correction", key: "fdr" },
                    { label: "Root cause taxonomy", key: "root" },
                    { label: "Guided remediation", key: "remed" },
                    { label: "Pre-deployment intervention", key: "prevention" },
                  ].map((row) => (
                    <tr key={row.key} className="border-b border-hairline last:border-0">
                      <td className="py-3.5 pl-5 pr-3 text-ink-muted">{row.label}</td>
                      {COMPETITORS.map((c) => {
                        const v = c[row.key as keyof typeof c];
                        return (
                          <td key={c.name} className="py-3.5 px-3 text-center">
                            {v === true && <CheckMark accent={c.name === "Nexora"} />}
                            {v === false && <span className="text-ink-faint">—</span>}
                            {v === "partial" && <span className="text-2xs font-mono text-ink-dim uppercase tracking-wider">partial</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-24">
          <div className="text-center max-w-3xl mx-auto">
            <h2 className="font-serif text-3xl sm:text-[64px] leading-[1.02] tracking-tight mb-6">
              The fence at the top of the cliff,
              <br />
              <span className="italic text-ink-muted">not the ambulance at the bottom.</span>
            </h2>
            <p className="text-md text-ink-muted max-w-xl mx-auto mb-10">
              Walk through a complete COMPAS audit — five models, full statistics, three reports.
            </p>
            <Link href="/audit">
              <Button variant="primary" size="md">
                Start a fairness audit
                <ArrowRight />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      <footer className="border-t border-hairline">
        <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-8 flex flex-wrap items-center justify-between gap-3 text-xs text-ink-dim">
          <div className="flex items-center gap-3">
            <span>© 2026 Nexora</span>
            <span className="text-ink-faint">·</span>
            <span>HackDavis 2026 · Best Statistical Model · Best Social Good</span>
          </div>
          <div className="flex items-center gap-4">
            <a className="hover:text-ink transition-colors" href="#">Methodology</a>
            <a className="hover:text-ink transition-colors" href="#">GitHub</a>
            <a className="hover:text-ink transition-colors" href="#">Contact</a>
          </div>
        </div>
      </footer>
    </main>
  );
}

function ArrowRight() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <path d="M3 7H11M11 7L7.5 3.5M11 7L7.5 10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function CheckMark({ accent }: { accent?: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" className={`inline ${accent ? "text-accent" : "text-success"}`}>
      <path d="M3 7L6 10L11 4" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}
