"use client";

import { useState } from "react";
import { Nav } from "@/components/nav";
import { StageRail, type RailStage } from "@/components/audit/stage-rail";
import { UploadStage } from "@/components/audit/stage-upload";
import { ConfigureStage, type AuditConfig } from "@/components/audit/stage-configure";
import { RunningStage, type StageData } from "@/components/audit/stage-running";
import { ResultsStage } from "@/components/audit/stage-results";
import type { ParsedCsv } from "@/lib/csv";

export default function AuditPage() {
  const [stage, setStage] = useState<RailStage>("upload");
  const [csv, setCsv] = useState<ParsedCsv | null>(null);
  const [cfg, setCfg] = useState<AuditConfig | null>(null);
  const [elapsedSec, setElapsedSec] = useState("0.0");
  const [finalData, setFinalData] = useState<StageData>({});

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

  // Opening-page rule: on the upload stage, the centered animated logo is the
  // first element on the page — no nav, no stage rail above it. Every other
  // stage keeps the standard chrome (Nav with logo on the left, StageRail).
  const isOpeningPage = stage === "upload";

  return (
    <main className="min-h-screen">
      {!isOpeningPage && <Nav context={navContext} status={navStatus} />}
      {!isOpeningPage && <StageRail active={stage} />}

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
            onComplete={(totalSec, data) => {
              setElapsedSec(totalSec.toFixed(1));
              setFinalData(data);
              setStage("results");
            }}
          />
        )}

        {stage === "results" && csv && cfg && (
          <ResultsStage
            csv={csv}
            cfg={cfg}
            elapsedSec={elapsedSec}
            stage7={finalData.stage7}
            stage8={finalData.stage8}
            onRestart={() => {
              setCsv(null);
              setCfg(null);
              setElapsedSec("0.0");
              setFinalData({});
              setStage("upload");
            }}
          />
        )}
      </div>
    </main>
  );
}
