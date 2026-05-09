"use client";

import { useState } from "react";
import { cn } from "@/lib/cn";
import { Badge } from "./primitives";

type Tab = "engineer" | "regulator" | "community";

const TABS: { key: Tab; label: string; meta: string }[] = [
  { key: "engineer", label: "Engineer", meta: "ML deployment scorecard" },
  { key: "regulator", label: "Regulator", meta: "EU AI Act · Article 9" },
  { key: "community", label: "Community", meta: "Plain-language · audio EN/ES" },
];

export function ReportTabs() {
  const [tab, setTab] = useState<Tab>("engineer");

  return (
    <div>
      <div role="tablist" aria-label="Audience reports" className="flex border-b border-hairline">
        {TABS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tab === t.key}
            onClick={() => setTab(t.key)}
            className={cn(
              "relative px-5 py-4 text-left transition-colors group",
              tab === t.key ? "text-ink" : "text-ink-muted hover:text-ink"
            )}
          >
            <div className="text-sm font-medium">{t.label}</div>
            <div className="text-xs text-ink-dim mt-0.5">{t.meta}</div>
            {tab === t.key && (
              <span className="absolute bottom-0 left-0 right-0 h-px bg-ink" />
            )}
          </button>
        ))}
        <div className="ml-auto self-center pr-5">
          <button className="text-xs text-ink-muted hover:text-ink inline-flex items-center gap-1.5 font-mono uppercase tracking-wider">
            Export
            <svg width="10" height="10" viewBox="0 0 10 10"><path d="M2 5L5 8L8 5M5 8V0" stroke="currentColor" strokeWidth="1.25" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>
          </button>
        </div>
      </div>

      <div className="p-6 sm:p-8 animate-fade-in" key={tab}>
        {tab === "engineer" && <EngineerReport />}
        {tab === "regulator" && <RegulatorReport />}
        {tab === "community" && <CommunityReport />}
      </div>
    </div>
  );
}

function EngineerReport() {
  return (
    <div className="grid lg:grid-cols-[1fr_320px] gap-8">
      <div className="space-y-5 max-w-[68ch]">
        <div className="flex items-center gap-2">
          <Badge tone="accent" dot>Recommendation</Badge>
          <span className="text-xs text-ink-dim font-mono uppercase tracking-wider">
            Confidence: high · Pareto-validated
          </span>
        </div>
        <p className="text-md text-ink leading-relaxed">
          Deploy <strong className="text-ink">Logistic Regression</strong>. It is the only model on the
          Pareto frontier whose equalized odds gap CI does not overlap the regulatory caution band
          (≥ 0.08).
        </p>
        <p className="text-sm text-ink-muted leading-relaxed">
          XGBoost achieves 1.8 pp higher AUC (0.890 vs 0.872) but its FPR disparity for African-American
          defendants is significantly worse — Dietterich 5×2 CV paired t-test, p = 0.007. Random Forest
          and Linear SVM are Pareto-dominated and excluded from consideration.
        </p>

        <div className="rounded-md border border-hairline bg-elevated/40 p-4 mt-6">
          <div className="text-2xs uppercase tracking-[0.16em] text-ink-dim mb-3">Pre-deployment checklist</div>
          <ul className="space-y-2.5 text-sm">
            {[
              "Per-group bootstrap CIs computed on every fairness metric (n=2,000)",
              "BH FDR correction applied across 5 models × 7 metrics × 2 groups (q=0.05)",
              "DeLong AUC test and McNemar paired-prediction test passed",
              "Pareto frontier verified — no recommended model is dominated",
              "SHAP cross-group analysis surfaces priors_count as the proxy mechanism",
            ].map((line) => (
              <li key={line} className="flex items-start gap-3">
                <CheckGlyph />
                <span className="text-ink-muted">{line}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <aside className="rounded-md border border-hairline bg-elevated/40 p-5 h-fit">
        <div className="text-2xs uppercase tracking-[0.16em] text-ink-dim mb-3">Statistical notes</div>
        <dl className="space-y-3 text-sm">
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">Bootstrap iterations</dt>
            <dd className="font-mono tabular">2,000</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">CV protocol</dt>
            <dd className="font-mono tabular">5×2 stratified</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">FDR threshold</dt>
            <dd className="font-mono tabular">q = 0.05</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">Min subgroup N</dt>
            <dd className="font-mono tabular">2,103</dd>
          </div>
          <div className="flex justify-between gap-4">
            <dt className="text-ink-muted">Power · MDE 5%</dt>
            <dd className="font-mono tabular text-success">0.94</dd>
          </div>
        </dl>
      </aside>
    </div>
  );
}

function RegulatorReport() {
  return (
    <div className="max-w-[72ch] space-y-6">
      <div className="flex items-center gap-2">
        <Badge tone="neutral">EU AI Act · Annex III · High-Risk</Badge>
        <Badge tone="neutral">ISO/IEC 42001</Badge>
      </div>

      <div className="space-y-4">
        <Section
          n="9.2"
          title="Risk management system"
          body="Five model architectures evaluated under controlled conditions. Subgroup harm modeled across the protected attribute (race) for both binary outcomes. Bootstrap-derived uncertainty intervals attached to every metric."
        />
        <Section
          n="9.3"
          title="Identification of foreseeable risk"
          body="Disparate FPR identified for the African-American subgroup across all candidates. Magnitude varies from 0.041 (Logistic Regression, recommended) to 0.118 (XGBoost, rejected). Differences confirmed via Dietterich 5×2 CV at p < 0.05."
        />
        <Section
          n="9.4"
          title="Risk mitigation measures"
          body="Threshold-based mitigation explicitly blocked by the remediation engine — root cause is proxy discrimination, not threshold misalignment. Mitigation path: feature decorrelation of priors_count from race, validated by SHAP rank permutation test (p = 0.003)."
        />
        <Section
          n="9.5"
          title="Residual risk disclosure"
          body="After deployment of the recommended model, residual disparate impact ratio is 0.917 (above the 4/5ths threshold of 0.80). Continued monitoring on a 30-day cadence via the longitudinal audit log."
        />
      </div>
    </div>
  );
}

function Section({ n, title, body }: { n: string; title: string; body: string }) {
  return (
    <div className="grid grid-cols-[80px_1fr] gap-4 pb-4 border-b border-hairline last:border-0">
      <div className="font-mono text-2xs text-ink-dim tracking-wider mt-1">§ {n}</div>
      <div>
        <h4 className="text-sm font-medium text-ink mb-1.5">{title}</h4>
        <p className="text-sm text-ink-muted leading-relaxed">{body}</p>
      </div>
    </div>
  );
}

function CommunityReport() {
  return (
    <div className="max-w-[64ch] space-y-7">
      <div className="flex items-center gap-3">
        <Badge tone="neutral">Plain language · FK grade 7.4</Badge>
        <button className="inline-flex items-center gap-2 text-xs text-ink-muted hover:text-ink transition-colors">
          <PlayGlyph />
          Listen · English
        </button>
        <button className="inline-flex items-center gap-2 text-xs text-ink-muted hover:text-ink transition-colors">
          <PlayGlyph />
          Escuchar · Español
        </button>
      </div>

      <p className="font-serif text-2xl leading-[1.25] text-ink">
        Of the five models we tested, the one we recommend treats Black defendants and white defendants the most
        equally — but it still does not treat them equally.
      </p>

      <div className="space-y-4 text-md text-ink-muted leading-relaxed">
        <p>
          Under the recommended model, a Black defendant is roughly <strong className="text-ink">4 percentage points</strong> more
          likely to be flagged as high-risk than a white defendant with similar history. Under the worst model we tested,
          that gap was <strong className="text-ink">12 percentage points</strong>.
        </p>
        <p>
          In real terms, choosing the recommended model means <strong className="text-ink">1,847 people</strong> in this dataset
          would receive a fairer decision than they would under the worst alternative.
        </p>
        <p>
          If you believe an algorithm has treated you unfairly, you can file a complaint with the agency that deployed it,
          or with your state attorney general. You have the right to ask which model was used to decide your case.
        </p>
      </div>
    </div>
  );
}

function CheckGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" className="mt-0.5 shrink-0 text-success" aria-hidden="true">
      <circle cx="7" cy="7" r="6.5" fill="currentColor" opacity="0.14" stroke="currentColor" strokeOpacity="0.3" />
      <path d="M4.5 7L6.25 8.75L9.5 5.25" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function PlayGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" className="text-ink-muted">
      <circle cx="7" cy="7" r="6.5" fill="none" stroke="currentColor" strokeOpacity="0.4" />
      <path d="M5.5 4.5L9.5 7L5.5 9.5V4.5Z" fill="currentColor" />
    </svg>
  );
}
