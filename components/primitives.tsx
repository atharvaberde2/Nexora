import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

/* ─────────────  Logo  ───────────── */

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <path d="M9 1.5v15M2 4.5h14M4.5 4.5L2 10.5h5L4.5 4.5zM13.5 4.5L11 10.5h5l-2.5-6z" stroke="#F5F6F7" strokeWidth="1.25" strokeLinejoin="round" fill="none" />
        <circle cx="4.5" cy="14.5" r="1.5" fill="#5B7FFF" />
      </svg>
      <span className="font-medium tracking-tight text-[15px]">Nexora</span>
    </div>
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
