"use client";

import { useCallback, useRef, useState } from "react";
import { Button, Card } from "@/components/primitives";
import { cn } from "@/lib/cn";
import {
  formatBytes,
  mergeInspectColumns,
  parseCsv,
  parseCsvText,
  SAMPLE_CSV,
  type ParsedCsv,
} from "@/lib/csv";
import { inspectCsv } from "@/lib/api";

export function UploadStage({
  onParsed,
}: {
  onParsed: (csv: ParsedCsv) => void;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const busy = busyLabel !== null;

  const inspectAndForward = useCallback(
    async (parsed: ParsedCsv) => {
      setBusyLabel("Inspecting columns…");
      try {
        const inspect = await inspectCsv(parsed);
        onParsed(mergeInspectColumns(parsed, inspect.columns));
      } catch (e) {
        // Backend down or unreachable — fall back to JS-detected types.
        console.warn("inspectCsv failed, using client-side types:", e);
        onParsed(parsed);
      }
    },
    [onParsed]
  );

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setBusyLabel("Parsing…");
      try {
        if (!file.name.toLowerCase().endsWith(".csv")) {
          throw new Error("Please upload a .csv file.");
        }
        if (file.size > 50 * 1024 * 1024) {
          throw new Error("File is larger than 50 MB. Use a sample for the demo.");
        }
        const parsed = await parseCsv(file);
        await inspectAndForward(parsed);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to parse CSV.");
        setBusyLabel(null);
      }
    },
    [inspectAndForward]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const useSample = useCallback(async () => {
    setError(null);
    const parsed = parseCsvText(SAMPLE_CSV, "compas-sample.csv", SAMPLE_CSV.length);
    await inspectAndForward(parsed);
  }, [inspectAndForward]);

  return (
    <div className="max-w-[1320px] mx-auto px-5 sm:px-8 py-12 sm:py-16">
      <div className="grid lg:grid-cols-12 gap-12 items-start">
        {/* Left: copy */}
        <div className="lg:col-span-5">
          <div className="text-2xs uppercase tracking-[0.18em] text-ink-dim mb-4">
            Step 1 — Upload
          </div>
          <h1 className="font-serif text-3xl leading-[1.05] tracking-tight text-ink mb-5">
            Bring the dataset you'd train on.
          </h1>
          <p className="text-md text-ink-muted leading-relaxed mb-8">
            Nexora runs entirely on the data you upload — no records leave
            your browser during configuration. We need at least one column
            you'd predict, and one demographic attribute you want to audit
            fairness against.
          </p>

          <ul className="space-y-3 text-sm">
            {[
              { kbd: "CSV", text: "Comma-delimited, with a header row" },
              { kbd: "≥30", text: "Observations per protected group, minimum" },
              { kbd: "Binary", text: "Target column (positive / negative outcome)" },
            ].map((r) => (
              <li key={r.kbd} className="flex items-start gap-3">
                <span className="font-mono text-2xs uppercase tracking-wider text-ink-dim border border-hairline rounded px-1.5 h-5 inline-flex items-center mt-0.5">
                  {r.kbd}
                </span>
                <span className="text-ink-muted">{r.text}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Right: dropzone */}
        <div className="lg:col-span-7">
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />

          <Card
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={cn(
              "p-10 sm:p-14 text-center transition-all relative cursor-pointer",
              dragOver && "border-accent bg-accent-soft/40 shadow-glow",
              !dragOver && "hover:border-line"
            )}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
          >
            <div className="absolute inset-0 grid-bg opacity-30 pointer-events-none rounded-lg" />

            <div className="relative">
              <div className="mx-auto w-12 h-12 rounded-full border border-line bg-elevated flex items-center justify-center mb-6">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 20 20"
                  className={cn(
                    "transition-colors",
                    dragOver ? "text-accent" : "text-ink-muted"
                  )}
                  aria-hidden="true"
                >
                  <path
                    d="M10 3v10M5 8l5-5 5 5M3 17h14"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                  />
                </svg>
              </div>

              <div className="font-serif text-xl text-ink mb-2">
                {busyLabel ?? "Drop your CSV here"}
              </div>
              <div className="text-sm text-ink-muted mb-6">
                or click to browse · max 50 MB
              </div>

              <div className="flex items-center justify-center gap-3">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    inputRef.current?.click();
                  }}
                  disabled={busy}
                >
                  Choose file
                </Button>
                <span className="text-ink-faint text-xs">·</span>
                <button
                  className="text-xs text-ink-muted hover:text-ink transition-colors underline-offset-4 hover:underline"
                  onClick={(e) => {
                    e.stopPropagation();
                    useSample();
                  }}
                  disabled={busy}
                >
                  Use COMPAS sample
                </button>
              </div>

              {error && (
                <div className="mt-6 text-xs text-danger font-mono">{error}</div>
              )}
            </div>
          </Card>

          <div className="grid grid-cols-3 gap-2 mt-3 text-2xs font-mono text-ink-dim">
            <Format label="UTF-8" />
            <Format label="Quoted values OK" />
            <Format label="No row limit" />
          </div>
        </div>
      </div>
    </div>
  );
}

function Format({ label }: { label: string }) {
  return (
    <div className="border border-hairline rounded-md px-3 py-2.5 flex items-center justify-center gap-2">
      <span className="w-1 h-1 rounded-full bg-success" />
      <span className="uppercase tracking-wider">{label}</span>
    </div>
  );
}
