"use client";

import { useState } from "react";
import { MODELS, type Model } from "@/lib/data";
import { cn } from "@/lib/cn";

const W = 760;
const H = 460;
const PAD = { left: 72, right: 32, top: 32, bottom: 64 };

// Domain — keep slack on each side so ellipses don't clip
const X_DOMAIN: [number, number] = [0.025, 0.135];
const Y_DOMAIN: [number, number] = [0.844, 0.898];

function xScale(v: number) {
  const t = (v - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0]);
  return PAD.left + t * (W - PAD.left - PAD.right);
}
function yScale(v: number) {
  const t = (v - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0]);
  return H - PAD.bottom - t * (H - PAD.top - PAD.bottom);
}

const X_TICKS = [0.04, 0.06, 0.08, 0.10, 0.12];
const Y_TICKS = [0.85, 0.86, 0.87, 0.88, 0.89];

const COLOR_HEX: Record<string, string> = {
  m1: "#9CA3AF",
  m2: "#A78BFA",
  m3: "#F87171",
  m4: "#34D399",
  m5: "#FBBF24",
};

export function ParetoChart() {
  const [hover, setHover] = useState<Model | null>(null);

  // Build frontier from Pareto-optimal models, sorted by x
  const frontier = MODELS
    .filter((m) => m.paretoOptimal)
    .sort((a, b) => a.eqOddsGap - b.eqOddsGap);

  const frontierPath = frontier
    .map((m, i) => {
      const x = xScale(m.eqOddsGap);
      const y = yScale(m.auc);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-auto select-none"
        role="img"
        aria-label="Accuracy versus fairness Pareto frontier across five model families"
      >
        <defs>
          <radialGradient id="recGlow" cx="50%" cy="50%">
            <stop offset="0%" stopColor="#5B7FFF" stopOpacity="0.55" />
            <stop offset="60%" stopColor="#5B7FFF" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#5B7FFF" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="fairGrad" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#4ADE80" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#F87171" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Fairness gradient zone hint along x */}
        <rect
          x={PAD.left}
          y={PAD.top}
          width={W - PAD.left - PAD.right}
          height={H - PAD.top - PAD.bottom}
          fill="url(#fairGrad)"
          opacity={0.6}
        />

        {/* Gridlines */}
        {X_TICKS.map((t) => (
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
        {Y_TICKS.map((t) => (
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

        {/* Axis labels and ticks */}
        {X_TICKS.map((t) => (
          <text
            key={`tx-${t}`}
            x={xScale(t)}
            y={H - PAD.bottom + 22}
            textAnchor="middle"
            className="fill-ink-dim font-mono"
            fontSize={11}
          >
            {t.toFixed(2)}
          </text>
        ))}
        {Y_TICKS.map((t) => (
          <text
            key={`ty-${t}`}
            x={PAD.left - 14}
            y={yScale(t) + 4}
            textAnchor="end"
            className="fill-ink-dim font-mono"
            fontSize={11}
          >
            {t.toFixed(2)}
          </text>
        ))}

        {/* Axis titles */}
        <text
          x={PAD.left + (W - PAD.left - PAD.right) / 2}
          y={H - 18}
          textAnchor="middle"
          className="fill-ink-muted"
          fontSize={12}
          letterSpacing={0.4}
        >
          ← FAIRER       Equalized odds gap       LESS FAIR →
        </text>
        <text
          transform={`translate(${PAD.left - 50}, ${PAD.top + (H - PAD.top - PAD.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fill-ink-muted"
          fontSize={12}
          letterSpacing={0.4}
        >
          MORE ACCURATE →     AUC-ROC
        </text>

        {/* Pareto frontier line */}
        <path
          d={frontierPath}
          fill="none"
          stroke="#3A4150"
          strokeWidth={1.25}
          strokeDasharray="4 4"
          className="animate-fade-in"
          style={{ animationDelay: "200ms" }}
        />

        {/* Bootstrap ellipses */}
        {MODELS.map((m, i) => {
          const cx = xScale(m.eqOddsGap);
          const cy = yScale(m.auc);
          const dim = !m.paretoOptimal && !hover;
          return (
            <g
              key={`ell-${m.key}`}
              style={{ animationDelay: `${i * 80}ms` }}
              className="animate-fade-in"
            >
              <ellipse
                cx={cx}
                cy={cy}
                rx={m.ellipse.rx}
                ry={m.ellipse.ry}
                transform={`rotate(${m.ellipse.rot} ${cx} ${cy})`}
                fill={COLOR_HEX[m.color]}
                opacity={dim ? 0.06 : 0.14}
              />
              <ellipse
                cx={cx}
                cy={cy}
                rx={m.ellipse.rx}
                ry={m.ellipse.ry}
                transform={`rotate(${m.ellipse.rot} ${cx} ${cy})`}
                fill="none"
                stroke={COLOR_HEX[m.color]}
                strokeOpacity={dim ? 0.2 : 0.5}
                strokeWidth={1}
              />
            </g>
          );
        })}

        {/* Recommended glow */}
        {MODELS.filter((m) => m.recommended).map((m) => (
          <circle
            key={`glow-${m.key}`}
            cx={xScale(m.eqOddsGap)}
            cy={yScale(m.auc)}
            r={32}
            fill="url(#recGlow)"
          />
        ))}

        {/* Model points + labels */}
        {MODELS.map((m, i) => {
          const cx = xScale(m.eqOddsGap);
          const cy = yScale(m.auc);
          const isRec = m.recommended;
          const dominated = !m.paretoOptimal;
          const labelOffset = isRec ? -16 : -14;

          return (
            <g
              key={m.key}
              className="cursor-pointer animate-scale-in"
              style={{ animationDelay: `${300 + i * 80}ms` }}
              onMouseEnter={() => setHover(m)}
              onMouseLeave={() => setHover(null)}
            >
              {/* Hit target */}
              <circle cx={cx} cy={cy} r={22} fill="transparent" />
              {/* Outer ring for recommended */}
              {isRec && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={9}
                  fill="none"
                  stroke="#5B7FFF"
                  strokeWidth={1.5}
                  opacity={0.5}
                  className="origin-center"
                  style={{ transformOrigin: `${cx}px ${cy}px` }}
                />
              )}
              <circle
                cx={cx}
                cy={cy}
                r={isRec ? 5.5 : 4.5}
                fill={isRec ? "#5B7FFF" : COLOR_HEX[m.color]}
                opacity={dominated ? 0.55 : 1}
                stroke={isRec ? "#FFFFFF" : "#08090B"}
                strokeWidth={isRec ? 0 : 2}
              />
              <text
                x={cx + 12}
                y={cy + labelOffset}
                className={cn(
                  "font-medium",
                  dominated ? "fill-ink-dim" : "fill-ink"
                )}
                fontSize={12}
              >
                {m.name}
              </text>
              {dominated && (
                <text
                  x={cx + 12}
                  y={cy + labelOffset + 14}
                  className="fill-ink-dim font-mono"
                  fontSize={10}
                  letterSpacing={0.3}
                >
                  PARETO-DOMINATED
                </text>
              )}
              {isRec && (
                <text
                  x={cx + 12}
                  y={cy + labelOffset + 14}
                  className="fill-accent font-mono"
                  fontSize={10}
                  letterSpacing={0.4}
                >
                  ● RECOMMENDED
                </text>
              )}
            </g>
          );
        })}

        {/* Bottom-left "fairer is better" callout */}
        <g transform={`translate(${PAD.left + 14}, ${PAD.top + 14})`}>
          <rect width={148} height={22} rx={4} fill="#0F1115" stroke="#1F232B" />
          <circle cx={10} cy={11} r={3} fill="#4ADE80" />
          <text x={20} y={15} className="fill-ink-muted font-mono" fontSize={10} letterSpacing={0.4}>
            BETTER REGION (↑ AUC, ↓ GAP)
          </text>
        </g>
      </svg>

      {/* Hover detail card */}
      <div
        className={cn(
          "pointer-events-none absolute top-3 right-3 w-72 rounded-md border border-hairline bg-elevated/95 backdrop-blur p-4 transition-all duration-200",
          hover ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-1"
        )}
      >
        {hover && (
          <>
            <div className="flex items-center gap-2 mb-3">
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: COLOR_HEX[hover.color] }}
              />
              <span className="text-sm font-medium">{hover.name}</span>
              {hover.recommended && (
                <span className="ml-auto text-2xs font-mono tracking-wider text-accent">RECOMMENDED</span>
              )}
              {!hover.paretoOptimal && (
                <span className="ml-auto text-2xs font-mono tracking-wider text-ink-dim">DOMINATED</span>
              )}
            </div>
            <dl className="grid grid-cols-2 gap-y-2 gap-x-3 tabular text-xs">
              <dt className="text-ink-dim">AUC</dt>
              <dd className="text-right font-mono">
                {hover.auc.toFixed(3)}
                <span className="text-ink-dim ml-1">±{((hover.aucCi[1] - hover.aucCi[0]) / 2).toFixed(3)}</span>
              </dd>
              <dt className="text-ink-dim">Eq. odds gap</dt>
              <dd className="text-right font-mono">
                {hover.eqOddsGap.toFixed(3)}
                <span className="text-ink-dim ml-1">±{((hover.eqOddsCi[1] - hover.eqOddsCi[0]) / 2).toFixed(3)}</span>
              </dd>
              <dt className="text-ink-dim">FPR · Black</dt>
              <dd className="text-right font-mono">{hover.fprBlack.toFixed(3)}</dd>
              <dt className="text-ink-dim">FPR · White</dt>
              <dd className="text-right font-mono">{hover.fprWhite.toFixed(3)}</dd>
              <dt className="text-ink-dim">DeLong p</dt>
              <dd className="text-right font-mono">
                {hover.delongP < 0.05 ? (
                  <span className="text-warning">{hover.delongP.toFixed(3)}</span>
                ) : (
                  hover.delongP.toFixed(3)
                )}
              </dd>
            </dl>
          </>
        )}
      </div>
    </div>
  );
}
