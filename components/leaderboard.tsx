"use client";

import { useState } from "react";
import { MODELS, type Model } from "@/lib/data";
import { cn } from "@/lib/cn";
import { Badge } from "./primitives";

type Sort = { key: keyof Model; dir: "asc" | "desc" };

const COLOR_HEX: Record<string, string> = {
  m1: "#9CA3AF",
  m2: "#A78BFA",
  m3: "#F87171",
  m4: "#34D399",
  m5: "#FBBF24",
};

export function Leaderboard() {
  const [sort, setSort] = useState<Sort>({ key: "auc", dir: "desc" });

  const rows = [...MODELS].sort((a, b) => {
    const av = a[sort.key] as number;
    const bv = b[sort.key] as number;
    return sort.dir === "asc" ? av - bv : bv - av;
  });

  function toggle(key: keyof Model) {
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" }
    );
  }

  const cols: { key: keyof Model; label: string; align?: "right" }[] = [
    { key: "auc", label: "AUC", align: "right" },
    { key: "eqOddsGap", label: "Eq. odds Δ", align: "right" },
    { key: "dpGap", label: "Dem. parity Δ", align: "right" },
    { key: "ece", label: "ECE", align: "right" },
    { key: "fprBlack", label: "FPR · Black", align: "right" },
    { key: "fprWhite", label: "FPR · White", align: "right" },
    { key: "delongP", label: "DeLong p", align: "right" },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tabular">
        <thead>
          <tr className="text-2xs uppercase tracking-[0.16em] text-ink-dim border-b border-hairline">
            <th className="text-left font-normal py-3 pl-5 pr-3 w-[28%]">Model</th>
            {cols.map((c) => (
              <th
                key={String(c.key)}
                className={cn(
                  "font-normal py-3 px-3 cursor-pointer select-none hover:text-ink transition-colors",
                  c.align === "right" ? "text-right" : "text-left"
                )}
                onClick={() => toggle(c.key)}
              >
                <span className="inline-flex items-center gap-1">
                  {c.label}
                  <SortGlyph active={sort.key === c.key} dir={sort.dir} />
                </span>
              </th>
            ))}
            <th className="text-right font-normal py-3 pr-5 pl-3">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr
              key={m.key}
              className={cn(
                "border-b border-hairline last:border-0 transition-colors hover:bg-elevated/40",
                m.recommended && "bg-accent-soft/40"
              )}
            >
              <td className="py-3.5 pl-5 pr-3">
                <div className="flex items-center gap-3">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ background: COLOR_HEX[m.color] }}
                  />
                  <div>
                    <div className="font-medium">{m.name}</div>
                    <div className="text-xs text-ink-dim">{m.family}</div>
                  </div>
                </div>
              </td>
              <td className="py-3.5 px-3 text-right font-mono">
                {m.auc.toFixed(3)}
                <span className="text-ink-dim text-xs ml-1.5">
                  ±{((m.aucCi[1] - m.aucCi[0]) / 2).toFixed(3)}
                </span>
              </td>
              <td className="py-3.5 px-3 text-right font-mono">
                <span className={cn(m.eqOddsGap > 0.10 && "text-warning")}>
                  {m.eqOddsGap.toFixed(3)}
                </span>
                <span className="text-ink-dim text-xs ml-1.5">
                  ±{((m.eqOddsCi[1] - m.eqOddsCi[0]) / 2).toFixed(3)}
                </span>
              </td>
              <td className="py-3.5 px-3 text-right font-mono">{m.dpGap.toFixed(3)}</td>
              <td className="py-3.5 px-3 text-right font-mono">{m.ece.toFixed(3)}</td>
              <td className="py-3.5 px-3 text-right font-mono">{m.fprBlack.toFixed(3)}</td>
              <td className="py-3.5 px-3 text-right font-mono">{m.fprWhite.toFixed(3)}</td>
              <td className="py-3.5 px-3 text-right font-mono">
                <span className={cn(m.delongP < 0.01 && "text-danger", m.delongP >= 0.01 && m.delongP < 0.05 && "text-warning")}>
                  {m.delongP.toFixed(3)}
                </span>
              </td>
              <td className="py-3.5 pr-5 pl-3 text-right">
                {m.recommended ? (
                  <Badge tone="accent" dot>Recommended</Badge>
                ) : m.paretoOptimal ? (
                  <Badge tone="neutral">Pareto-opt</Badge>
                ) : (
                  <Badge tone="neutral" className="opacity-60">Dominated</Badge>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SortGlyph({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  return (
    <svg width="8" height="10" viewBox="0 0 8 10" className={cn(active ? "opacity-100" : "opacity-30")}>
      <path
        d="M4 0L8 4H0L4 0Z"
        fill="currentColor"
        opacity={active && dir === "asc" ? 1 : 0.4}
      />
      <path
        d="M4 10L0 6H8L4 10Z"
        fill="currentColor"
        opacity={active && dir === "desc" ? 1 : 0.4}
      />
    </svg>
  );
}
