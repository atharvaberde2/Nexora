import { cn } from "@/lib/cn";

export type RailStage = "upload" | "configure" | "running" | "results";

const STEPS: { key: RailStage; label: string; sub: string }[] = [
  { key: "upload", label: "Upload", sub: "Bring your dataset" },
  { key: "configure", label: "Configure", sub: "Target & protected attribute" },
  { key: "running", label: "Audit", sub: "Eight-stage pipeline" },
  { key: "results", label: "Review", sub: "Results & reports" },
];

export function StageRail({ active }: { active: RailStage }) {
  const activeIdx = STEPS.findIndex((s) => s.key === active);

  return (
    <div className="border-b border-hairline bg-canvas/60 backdrop-blur-sm">
      <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-5">
        <ol className="grid grid-cols-4 gap-2 sm:gap-6">
          {STEPS.map((s, i) => {
            const state =
              i < activeIdx ? "complete" : i === activeIdx ? "active" : "pending";
            return (
              <li
                key={s.key}
                className={cn(
                  "flex items-start gap-3 transition-opacity",
                  state === "pending" && "opacity-50"
                )}
              >
                <StepGlyph index={i + 1} state={state} />
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "text-xs sm:text-sm font-medium",
                        state === "active" && "text-ink",
                        state === "complete" && "text-ink",
                        state === "pending" && "text-ink-muted"
                      )}
                    >
                      {s.label}
                    </span>
                    {state === "active" && (
                      <span className="hidden sm:inline-block w-1 h-1 rounded-full bg-accent animate-pulse" />
                    )}
                  </div>
                  <div className="text-2xs text-ink-dim hidden sm:block mt-0.5 truncate">
                    {s.sub}
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

function StepGlyph({
  index,
  state,
}: {
  index: number;
  state: "complete" | "active" | "pending";
}) {
  return (
    <div
      className={cn(
        "shrink-0 w-7 h-7 rounded-full flex items-center justify-center font-mono text-xs transition-all",
        state === "complete" && "bg-ink text-canvas",
        state === "active" && "bg-accent text-canvas shadow-[0_0_0_4px_rgba(91,127,255,0.16)]",
        state === "pending" && "border border-line text-ink-dim"
      )}
    >
      {state === "complete" ? (
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <path
            d="M3 6L5 8L9 4"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            fill="none"
          />
        </svg>
      ) : (
        <span className="tabular">{index}</span>
      )}
    </div>
  );
}
