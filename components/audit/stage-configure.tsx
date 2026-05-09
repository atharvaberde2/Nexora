"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge, Button, Card } from "@/components/primitives";
import { cn } from "@/lib/cn";
import { formatBytes, type Column, type ParsedCsv } from "@/lib/csv";

export type AuditConfig = {
  target: string;
  positiveClass: string;
  protectedAttrs: string[];
};

const MAX_PROTECTED_ATTRS = 4;

export function ConfigureStage({
  csv,
  onBack,
  onConfirm,
}: {
  csv: ParsedCsv;
  onBack: () => void;
  onConfirm: (cfg: AuditConfig) => void;
}) {
  // Auto-suggest defaults
  const suggestedTarget = useMemo(
    () =>
      csv.columns.find((c) => c.type === "binary")?.name ||
      csv.columns.find((c) => /target|label|outcome|recid|approved/i.test(c.name))?.name ||
      "",
    [csv]
  );
  const suggestedProtected = useMemo(
    () =>
      csv.columns.find((c) =>
        /race|sex|gender|ethnic|age_group|disability/i.test(c.name)
      )?.name ||
      csv.columns.find((c) => c.type === "categorical" && c.unique <= 6)?.name ||
      "",
    [csv]
  );

  const [target, setTarget] = useState(suggestedTarget);
  const [protectedAttrs, setProtectedAttrs] = useState<string[]>(
    suggestedProtected ? [suggestedProtected] : []
  );
  const [positiveClass, setPositiveClass] = useState<string>("");

  const targetCol = csv.columns.find((c) => c.name === target);
  const protectedCols = useMemo(
    () => protectedAttrs
      .map((n) => csv.columns.find((c) => c.name === n))
      .filter((c): c is Column => Boolean(c)),
    [protectedAttrs, csv.columns]
  );

  // Auto-pick positive class when target changes
  useEffect(() => {
    if (!targetCol) return;
    if (targetCol.type === "binary" && targetCol.uniqueValues) {
      const v =
        targetCol.uniqueValues.find((x) => /^(1|true|yes|positive|approved|y)$/i.test(x)) ||
        targetCol.uniqueValues[0];
      setPositiveClass(v);
    } else if (targetCol.type === "numeric") {
      setPositiveClass("1");
    }
  }, [targetCol?.name, targetCol?.type, targetCol?.uniqueValues?.join(",")]);

  const groupSizesByAttr = useMemo(() => {
    return protectedCols.map((col) => {
      const counts = new Map<string, number>();
      for (const r of csv.rows) {
        const v = r[col.index];
        if (!v) continue;
        counts.set(v, (counts.get(v) || 0) + 1);
      }
      const groups = Array.from(counts.entries())
        .map(([name, n]) => ({ name, n }))
        .sort((a, b) => b.n - a.n);
      return { col, groups };
    });
  }, [protectedCols, csv]);

  function toggleProtected(name: string) {
    setProtectedAttrs((prev) => {
      if (prev.includes(name)) return prev.filter((p) => p !== name);
      if (prev.length >= MAX_PROTECTED_ATTRS) return prev;
      return [...prev, name];
    });
  }

  const warnings: string[] = [];
  if (!target) warnings.push("Pick a target column");
  if (protectedAttrs.length === 0) warnings.push("Pick at least one protected attribute");
  if (target && protectedAttrs.includes(target))
    warnings.push("Target and protected attribute must differ");
  if (groupSizesByAttr.some(({ groups }) => groups.some((g) => g.n < 30)))
    warnings.push("Some groups have fewer than 30 observations — power will be low");

  const ready =
    target &&
    protectedAttrs.length > 0 &&
    positiveClass &&
    !protectedAttrs.includes(target);

  return (
    <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-12">
      {/* Dataset header */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-8">
        <div>
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-2">
            Step 2 — Configure
          </div>
          <h1 className="font-serif text-2xl leading-tight text-ink mb-2">
            Tell Nexora what to predict, and who to audit fairness for.
          </h1>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-muted font-mono">
            <span className="text-ink">{csv.fileName}</span>
            <span>{formatBytes(csv.fileSize)}</span>
            <span className="tabular">{csv.rowCount.toLocaleString()} rows</span>
            <span className="tabular">{csv.headers.length} columns</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={onBack}>
            Replace file
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-12 gap-5">
        {/* Column inspector */}
        <Card className="lg:col-span-7 p-0 overflow-hidden">
          <div className="px-5 pt-4 pb-3 border-b border-hairline">
            <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1">
              Detected columns
            </div>
            <div className="text-sm">
              {csv.columns.length} columns · types inferred via pandas
            </div>
          </div>
          <div className="max-h-[460px] overflow-y-auto">
            <table className="w-full text-sm tabular">
              <thead className="sticky top-0 bg-surface z-10">
                <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
                  <th className="text-left font-normal py-3 pl-5 pr-3">Column</th>
                  <th className="text-left font-normal py-3 px-3 hidden sm:table-cell">Type</th>
                  <th className="text-right font-normal py-3 px-3">Unique</th>
                  <th className="text-right font-normal py-3 px-3">Missing</th>
                  <th className="text-left font-normal py-3 px-5 hidden md:table-cell">Sample</th>
                </tr>
              </thead>
              <tbody>
                {csv.columns.map((c) => (
                  <ColumnRow
                    key={c.name}
                    column={c}
                    isTarget={c.name === target}
                    isProtected={protectedAttrs.includes(c.name)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Selection panel */}
        <div className="lg:col-span-5 space-y-4">
          <FieldSelect
            label="Target column"
            hint="The outcome you want to predict"
            value={target}
            onChange={setTarget}
            options={csv.columns}
            preferTypes={["binary", "numeric"]}
          />

          {targetCol && (
            <PositiveClassSelector
              column={targetCol}
              value={positiveClass}
              onChange={setPositiveClass}
            />
          )}

          <ProtectedMultiSelect
            columns={csv.columns}
            selected={protectedAttrs}
            onToggle={toggleProtected}
            target={target}
          />

          {groupSizesByAttr.length > 0 && (
            <Card className="p-4 space-y-5">
              <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
                Group distribution
              </div>
              {groupSizesByAttr.map(({ col, groups }) => {
                const total = groups.reduce((s, x) => s + x.n, 0) || 1;
                return (
                  <div key={col.name}>
                    <div className="text-2xs font-mono uppercase tracking-wider text-ink mb-2">
                      {col.name}
                    </div>
                    <div className="space-y-2">
                      {groups.slice(0, 8).map((g) => {
                        const pct = (g.n / total) * 100;
                        return (
                          <div key={g.name}>
                            <div className="flex items-baseline justify-between text-xs mb-1">
                              <span className="text-ink-muted">{g.name}</span>
                              <span className="font-mono tabular text-ink-dim">
                                {g.n.toLocaleString()}{" "}
                                <span className="text-ink-faint">· {pct.toFixed(1)}%</span>
                              </span>
                            </div>
                            <div className="h-1 rounded-full bg-elevated overflow-hidden">
                              <div
                                className={cn(
                                  "h-full transition-all",
                                  g.n < 30 ? "bg-warning" : "bg-accent"
                                )}
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                      {groups.length > 8 && (
                        <div className="text-2xs font-mono text-ink-dim">
                          + {groups.length - 8} more groups
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {groupSizesByAttr.some(({ groups }) => groups.some((g) => g.n < 30)) && (
                <div className="text-xs text-warning flex items-start gap-2">
                  <svg width="12" height="12" viewBox="0 0 12 12" className="mt-0.5 shrink-0">
                    <path
                      d="M6 1L11 10H1L6 1Z M6 5V7 M6 8.5V8.6"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      fill="none"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span>
                    Power may be low for groups under 30 observations. Nexora
                    will still run, but minimum detectable effect size will be wide.
                  </span>
                </div>
              )}
            </Card>
          )}

          {warnings.length > 0 && (
            <div className="text-xs text-ink-muted space-y-1">
              {warnings.map((w) => (
                <div key={w} className="flex items-center gap-2">
                  <span className="w-1 h-1 rounded-full bg-warning" />
                  {w}
                </div>
              ))}
            </div>
          )}

          <div className="flex items-center gap-3 pt-2">
            <Button variant="secondary" size="md" onClick={onBack}>
              Back
            </Button>
            <Button
              variant="primary"
              size="md"
              disabled={!ready}
              onClick={() =>
                onConfirm({
                  target,
                  protectedAttrs,
                  positiveClass,
                })
              }
            >
              Run audit
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
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ColumnRow({
  column,
  isTarget,
  isProtected,
}: {
  column: Column;
  isTarget: boolean;
  isProtected: boolean;
}) {
  return (
    <tr
      className={cn(
        "border-b border-hairline last:border-0 transition-colors",
        (isTarget || isProtected) && "bg-accent-soft/40"
      )}
    >
      <td className="py-3 pl-5 pr-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-xs truncate text-ink">{column.name}</span>
          {isTarget && (
            <Badge tone="accent" className="shrink-0">target</Badge>
          )}
          {isProtected && (
            <Badge tone="accent" className="shrink-0">protected</Badge>
          )}
        </div>
      </td>
      <td className="py-3 px-3 hidden sm:table-cell">
        <TypeBadge type={column.type} />
      </td>
      <td className="py-3 px-3 text-right font-mono text-ink-muted">
        {column.unique.toLocaleString()}
      </td>
      <td className="py-3 px-3 text-right font-mono">
        <span className={cn(column.missingPct > 0.1 && "text-warning")}>
          {(column.missingPct * 100).toFixed(1)}%
        </span>
      </td>
      <td className="py-3 pl-3 pr-5 hidden md:table-cell">
        <div className="text-xs text-ink-dim font-mono truncate max-w-[180px]">
          {column.sample.slice(0, 3).join(", ")}
          {column.sample.length > 3 && "…"}
        </div>
      </td>
    </tr>
  );
}

function TypeBadge({ type }: { type: Column["type"] }) {
  const map = {
    numeric: { label: "numeric", tone: "neutral" as const },
    binary: { label: "binary", tone: "neutral" as const },
    categorical: { label: "categorical", tone: "neutral" as const },
    freetext: { label: "freetext", tone: "neutral" as const },
  };
  return <Badge tone={map[type].tone}>{map[type].label}</Badge>;
}

function FieldSelect({
  label,
  hint,
  value,
  onChange,
  options,
  preferTypes,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
  options: Column[];
  preferTypes?: Column["type"][];
}) {
  const sorted = useMemo(() => {
    if (!preferTypes) return options;
    return [...options].sort((a, b) => {
      const aPref = preferTypes.includes(a.type) ? 0 : 1;
      const bPref = preferTypes.includes(b.type) ? 0 : 1;
      if (aPref !== bPref) return aPref - bPref;
      return a.name.localeCompare(b.name);
    });
  }, [options, preferTypes]);

  return (
    <Card className="p-4">
      <label className="block">
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">{label}</span>
          {hint && <span className="text-2xs text-ink-dim">{hint}</span>}
        </div>
        <div className="relative">
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="w-full bg-elevated border border-line rounded-md px-3 h-11 text-sm font-mono appearance-none focus:border-accent transition-colors hover:border-ink-dim"
          >
            <option value="">— select —</option>
            {sorted.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} · {c.type}
              </option>
            ))}
          </select>
          <svg
            width="10"
            height="10"
            viewBox="0 0 10 10"
            className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-dim pointer-events-none"
            aria-hidden="true"
          >
            <path d="M2 4L5 7L8 4" stroke="currentColor" strokeWidth="1.25" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </label>
    </Card>
  );
}

function ProtectedMultiSelect({
  columns,
  selected,
  onToggle,
  target,
}: {
  columns: Column[];
  selected: string[];
  onToggle: (name: string) => void;
  target: string;
}) {
  // Eligible: binary, categorical, OR low-cardinality numeric (≤ 10 unique).
  // Hide freetext + high-cardinality numeric — these are non-sensical as
  // protected attrs without binning.
  const eligible = useMemo(() => {
    return columns
      .filter((c) => {
        if (c.name === target) return false;
        if (c.type === "binary" || c.type === "categorical") return true;
        if (c.type === "numeric" && c.unique <= 10) return true;
        return false;
      })
      .sort((a, b) => {
        const aPref = a.type === "categorical" || a.type === "binary" ? 0 : 1;
        const bPref = b.type === "categorical" || b.type === "binary" ? 0 : 1;
        if (aPref !== bPref) return aPref - bPref;
        return a.name.localeCompare(b.name);
      });
  }, [columns, target]);

  return (
    <Card className="p-4">
      <div className="flex items-baseline justify-between mb-3">
        <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Protected attribute(s)
        </span>
        <span className="text-2xs text-ink-dim">
          {selected.length}/{MAX_PROTECTED_ATTRS} selected · audited independently
        </span>
      </div>
      {eligible.length === 0 ? (
        <div className="text-xs text-ink-muted">
          No eligible columns. Need binary, categorical, or low-cardinality
          numeric (≤ 10 unique).
        </div>
      ) : (
        <div className="max-h-56 overflow-y-auto space-y-1.5 -mx-1 px-1">
          {eligible.map((c) => {
            const isSelected = selected.includes(c.name);
            const atLimit = !isSelected && selected.length >= MAX_PROTECTED_ATTRS;
            return (
              <button
                key={c.name}
                type="button"
                disabled={atLimit}
                onClick={() => onToggle(c.name)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 h-10 rounded-md border transition-all text-left",
                  isSelected
                    ? "border-accent bg-accent-soft/40 text-ink"
                    : "border-line bg-elevated text-ink-muted hover:border-ink-dim hover:text-ink",
                  atLimit && "opacity-40 cursor-not-allowed hover:border-line"
                )}
              >
                <span
                  className={cn(
                    "w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-colors",
                    isSelected ? "border-accent bg-accent" : "border-line bg-canvas"
                  )}
                  aria-hidden="true"
                >
                  {isSelected && (
                    <svg width="10" height="10" viewBox="0 0 10 10">
                      <path
                        d="M2 5L4 7L8 3"
                        stroke="#08090B"
                        strokeWidth="1.6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        fill="none"
                      />
                    </svg>
                  )}
                </span>
                <span className="font-mono text-xs truncate flex-1">{c.name}</span>
                <span className="text-2xs font-mono text-ink-dim shrink-0">
                  {c.type} · {c.unique}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </Card>
  );
}

function PositiveClassSelector({
  column,
  value,
  onChange,
}: {
  column: Column;
  value: string;
  onChange: (v: string) => void;
}) {
  const values =
    column.uniqueValues ||
    (column.type === "numeric" ? ["1"] : []);

  if (values.length === 0) return null;

  return (
    <Card className="p-4">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-2xs uppercase tracking-[0.18em] text-ink-dim">
          Positive class
        </span>
        <span className="text-2xs text-ink-dim">Which value = "positive"?</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {values.slice(0, 8).map((v) => {
          const selected = v === value;
          return (
            <button
              key={v}
              onClick={() => onChange(v)}
              className={cn(
                "h-9 px-3 rounded-md border text-sm font-mono transition-all",
                selected
                  ? "border-accent bg-accent text-canvas font-medium"
                  : "border-line bg-elevated text-ink-muted hover:border-ink-dim hover:text-ink"
              )}
            >
              {v}
            </button>
          );
        })}
      </div>
    </Card>
  );
}
