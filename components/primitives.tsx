import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

/* ─────────────  Logo  ─────────────
 *
 * NEXORA wordmark with a balance-scale icon to the LEFT.
 * Scale ⚖ → fairness / ML audit / justice.
 *
 * Letter-by-letter assembly: each letter rises and fades in, staggered 70ms.
 * The animation runs once on mount via the CSS keyframes in globals.css.
 *
 * `size`: "sm" for nav (default, small), "lg" for the centered hero on the
 * opening page. Scale icon and text scale together; animation works for both.
 */

const NEXORA_LETTERS = ["N", "E", "X", "O", "R", "A"] as const;

export function Logo({
  className,
  size = "sm",
}: {
  className?: string;
  size?: "sm" | "lg";
}) {
  const isLg = size === "lg";
  const iconPx = isLg ? 36 : 18;
  const gap = isLg ? "gap-3" : "gap-2";
  const letterCls = isLg
    ? "font-serif text-[44px] sm:text-[56px] tracking-[-0.02em] leading-none"
    : "font-medium tracking-tight text-[15px] leading-none";

  return (
    <div className={cn("inline-flex items-center", gap, className)} aria-label="Nexora">
      {/* Balance scale — fairness symbol, baseline-aligned with the wordmark */}
      <ScaleIcon size={iconPx} className="logo-scale shrink-0" />
      <span className={cn(letterCls, "select-none")}>
        {NEXORA_LETTERS.map((char, i) => (
          <span
            key={i}
            className="logo-letter"
            style={{ animationDelay: `${120 + i * 70}ms` }}
          >
            {char}
          </span>
        ))}
      </span>
    </div>
  );
}

/** Two-pan balance scale — the fairness/justice mark. Baseline-aligned with
 *  the wordmark via viewBox bottom edge. Stroke uses currentColor so it picks
 *  up the text color of the parent. */
function ScaleIcon({ size, className }: { size: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={className}
      aria-hidden="true"
      fill="none"
    >
      {/* Vertical column */}
      <path d="M12 3.5V21" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      {/* Base */}
      <path d="M8 21H16" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      {/* Crossbeam */}
      <path d="M4 7H20" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      {/* Top knob */}
      <circle cx="12" cy="3.5" r="1" fill="currentColor" />
      {/* Left pan: triangle from (4,7) hanging by short threads */}
      <path d="M4 7L1.6 12.2H6.4L4 7Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      {/* Right pan */}
      <path d="M20 7L17.6 12.2H22.4L20 7Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      {/* Subtle accent under the right pan to suggest balance/equity */}
      <path d="M1.6 12.2H6.4" stroke="#738BF2" strokeWidth="1.3" strokeLinecap="round" opacity="0.85" />
      <path d="M17.6 12.2H22.4" stroke="#738BF2" strokeWidth="1.3" strokeLinecap="round" opacity="0.85" />
    </svg>
  );
}

/* ─────────────  Button  ───────────── */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", className, ...props },
  ref
) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-medium rounded-md transition-all duration-150 active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none",
        size === "sm" ? "h-8 px-3 text-xs" : "h-10 px-4 text-sm",
        variant === "primary" &&
          "bg-ink text-canvas hover:bg-white shadow-[0_1px_0_rgba(255,255,255,0.5)_inset]",
        variant === "secondary" &&
          "bg-elevated text-ink border border-line hover:border-ink-dim hover:bg-[#1B1F26]",
        variant === "ghost" && "text-ink-muted hover:text-ink hover:bg-elevated",
        className
      )}
      {...props}
    />
  );
});

/* ─────────────  Card  ───────────── */

type CardProps = HTMLAttributes<HTMLDivElement>;

export function Card({ className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-hairline bg-surface relative overflow-hidden",
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({
  eyebrow,
  title,
  meta,
  className,
}: {
  eyebrow?: string;
  title?: React.ReactNode;
  meta?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-end justify-between px-5 pt-4 pb-3 border-b border-hairline", className)}>
      <div>
        {eyebrow && (
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">{eyebrow}</div>
        )}
        {title && <div className="text-sm font-medium">{title}</div>}
      </div>
      {meta && <div className="text-2xs text-ink-dim font-mono tracking-wider">{meta}</div>}
    </div>
  );
}

/* ─────────────  Badge  ───────────── */

type Tone = "neutral" | "accent" | "success" | "warning" | "danger";
const toneClass: Record<Tone, string> = {
  neutral: "bg-elevated text-ink-muted border-line",
  accent: "bg-accent-soft text-accent border-accent/30",
  success: "bg-success/10 text-success border-success/20",
  warning: "bg-warning/10 text-warning border-warning/20",
  danger: "bg-danger/10 text-danger border-danger/25",
};

export function Badge({
  tone = "neutral",
  dot = false,
  children,
  className,
}: {
  tone?: Tone;
  dot?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 h-[22px] text-2xs font-mono uppercase tracking-[0.14em]",
        toneClass[tone],
        className
      )}
    >
      {dot && <span className="w-1.5 h-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}

/* ─────────────  Stat number with CI  ───────────── */

export function Stat({
  label,
  value,
  ci,
  unit,
  hint,
  size = "lg",
}: {
  label: string;
  value: string;
  ci?: string;
  unit?: string;
  hint?: string;
  size?: "md" | "lg" | "xl";
}) {
  return (
    <div>
      <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-1.5">{label}</div>
      <div
        className={cn(
          "font-mono tabular text-ink",
          size === "md" && "text-lg",
          size === "lg" && "text-xl",
          size === "xl" && "text-2xl"
        )}
      >
        {value}
        {unit && <span className="text-ink-muted text-sm ml-1 font-sans">{unit}</span>}
      </div>
      {ci && <div className="text-xs text-ink-dim font-mono mt-0.5">95% CI {ci}</div>}
      {hint && <div className="text-xs text-ink-muted mt-1">{hint}</div>}
    </div>
  );
}
