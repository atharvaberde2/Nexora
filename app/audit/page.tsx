"use client";

import { useState } from "react";
import { Nav } from "@/components/nav";
import { StageRail, type RailStage } from "@/components/audit/stage-rail";
import { UploadStage } from "@/components/audit/stage-upload";
import { ConfigureStage, type AuditConfig } from "@/components/audit/stage-configure";
import { RunningStage } from "@/components/audit/stage-running";
import { ResultsStage } from "@/components/audit/stage-results";
import type { ParsedCsv } from "@/lib/csv";

export default function AuditPage() {
  const [stage, setStage] = useState<RailStage>("upload");
  const [csv, setCsv] = useState<ParsedCsv | null>(null);
  const [cfg, setCfg] = useState<AuditConfig | null>(null);
  const [elapsedSec, setElapsedSec] = useState("0.0");

  const navContext = csv
    ? {
        kind: "AUDIT",
        name: csv.fileName.replace(/\.csv$/i, ""),
      }
    : { kind: "AUDIT", name: "new" };

  const navStatus =
    stage === "upload"
      ? "Awaiting data"
      : stage === "configure"
      ? "Configuring"
      : stage === "running"
      ? "Running"
      : "Audit complete";

  return (
    <main className="min-h-screen">
      <Nav context={navContext} status={navStatus} />
      <StageRail active={stage} />

      <div key={stage} className="animate-fade-up">
        {stage === "upload" && (
          <UploadStage
            onParsed={(parsed) => {
              setCsv(parsed);
              setStage("configure");
            }}
          />
        )}

        {stage === "configure" && csv && (
          <ConfigureStage
            csv={csv}
            onBack={() => {
              setCsv(null);
              setStage("upload");
            }}
            onConfirm={(config) => {
              setCfg(config);
              setStage("running");
            }}
          />
        )}

        {stage === "running" && csv && cfg && (
          <RunningStage
            csv={csv}
            cfg={cfg}
            onComplete={(totalSec) => {
              setElapsedSec(totalSec.toFixed(1));
              setStage("results");
            }}
          />
        )}

        {stage === "results" && csv && cfg && (
          <ResultsStage
            csv={csv}
            cfg={cfg}
            elapsedSec={elapsedSec}
            onRestart={() => {
              setCsv(null);
              setCfg(null);
              setElapsedSec("0.0");
              setStage("upload");
            }}
          />
        )}
      </div>
    </main>
  );
}
