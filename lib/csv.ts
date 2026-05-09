// Lightweight CSV parser + column inference. Client-side only.
// Handles quoted values, escaped quotes, CRLF, and very large files
// by sampling rows for type inference (whole file kept for row count).

export type ColumnType = "numeric" | "binary" | "categorical" | "freetext";

export type Column = {
  name: string;
  index: number;
  type: ColumnType;
  unique: number;
  uniqueValues?: string[];
  sample: string[];
  numericRange?: [number, number];
  missingPct: number;
};

export type ParsedCsv = {
  fileName: string;
  fileSize: number;
  headers: string[];
  rows: string[][];
  rowCount: number;
  columns: Column[];
  text: string;
};

const SAMPLE_LIMIT = 5000;

export async function parseCsv(file: File): Promise<ParsedCsv> {
  const text = await file.text();
  return parseCsvText(text, file.name, file.size);
}

export function parseCsvText(text: string, fileName: string, fileSize: number): ParsedCsv {
  // Strip BOM
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);

  const lines = splitLines(text);
  if (lines.length < 2) {
    throw new Error("CSV must have a header row and at least one data row.");
  }

  const headers = parseLine(lines[0]).map((h, i) => h.trim() || `column_${i}`);
  const rows = lines.slice(1).map(parseLine);

  // Sample for type inference
  const sample = rows.slice(0, SAMPLE_LIMIT);
  const columns = headers.map((name, i) => analyzeColumn(name, i, sample));

  return {
    fileName,
    fileSize,
    headers,
    rows,
    rowCount: rows.length,
    columns,
    text,
  };
}

function splitLines(text: string): string[] {
  // Split on \n, handle \r — but respect quoted newlines
  const lines: string[] = [];
  let current = "";
  let inQuote = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      // Toggle quote state, accounting for escaped quotes
      if (inQuote && text[i + 1] === '"') {
        current += '""';
        i++;
      } else {
        inQuote = !inQuote;
        current += c;
      }
    } else if ((c === "\n" || c === "\r") && !inQuote) {
      if (c === "\r" && text[i + 1] === "\n") i++;
      if (current.trim().length > 0) lines.push(current);
      current = "";
    } else {
      current += c;
    }
  }
  if (current.trim().length > 0) lines.push(current);
  return lines;
}

function parseLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQuote && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuote = !inQuote;
      }
    } else if (c === "," && !inQuote) {
      result.push(current);
      current = "";
    } else {
      current += c;
    }
  }
  result.push(current);
  return result.map((v) => v.trim());
}

function analyzeColumn(name: string, index: number, rows: string[][]): Column {
  const values = rows.map((r) => r[index] ?? "");
  const present = values.filter((v) => v !== "" && v.toLowerCase() !== "na" && v.toLowerCase() !== "null");
  const missingPct = (values.length - present.length) / Math.max(values.length, 1);

  const uniqueSet = new Set(present);
  const sample = Array.from(uniqueSet).slice(0, 6);

  // Try numeric — must be > 95% parseable to count
  const numerics: number[] = [];
  for (const v of present) {
    const n = Number(v);
    if (Number.isFinite(n)) numerics.push(n);
  }
  const isNumeric = present.length > 0 && numerics.length / present.length > 0.95;

  if (isNumeric) {
    return {
      name,
      index,
      type: "numeric",
      unique: uniqueSet.size,
      sample,
      numericRange: [Math.min(...numerics), Math.max(...numerics)],
      missingPct,
    };
  }

  if (uniqueSet.size === 2) {
    return {
      name,
      index,
      type: "binary",
      unique: 2,
      uniqueValues: Array.from(uniqueSet),
      sample,
      missingPct,
    };
  }

  if (uniqueSet.size <= 30) {
    return {
      name,
      index,
      type: "categorical",
      unique: uniqueSet.size,
      uniqueValues: Array.from(uniqueSet),
      sample,
      missingPct,
    };
  }

  return {
    name,
    index,
    type: "freetext",
    unique: uniqueSet.size,
    sample,
    missingPct,
  };
}

// Overlay backend-derived column metadata (pandas-authoritative types,
// unique counts, missingness) onto a client-parsed CSV. Columns the backend
// doesn't return are kept as-is.
export function mergeInspectColumns(
  parsed: ParsedCsv,
  backendCols: {
    name: string;
    type: ColumnType;
    unique: number;
    missing_pct: number;
    sample: string[];
    unique_values: string[] | null;
    numeric_range: [number, number] | null;
  }[]
): ParsedCsv {
  const byName = new Map(backendCols.map((c) => [c.name, c]));
  const merged = parsed.columns.map((c) => {
    const b = byName.get(c.name);
    if (!b) return c;
    return {
      ...c,
      type: b.type,
      unique: b.unique,
      missingPct: b.missing_pct,
      sample: b.sample.length > 0 ? b.sample : c.sample,
      uniqueValues: b.unique_values ?? c.uniqueValues,
      numericRange: b.numeric_range ?? c.numericRange,
    };
  });
  return { ...parsed, columns: merged };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

// A small synthetic COMPAS-like sample so users can demo without their own file.
export const SAMPLE_CSV = `id,age,sex,race,priors_count,charge_degree,decile_score,two_year_recid
1,69,Male,Other,0,F,1,0
2,34,Male,African-American,0,F,3,1
3,24,Male,African-American,4,F,4,1
4,23,Male,African-American,1,F,8,0
5,43,Male,Other,2,F,1,0
6,44,Male,Other,0,M,1,0
7,41,Male,Caucasian,14,F,6,1
8,43,Male,Other,3,F,4,0
9,39,Female,Caucasian,0,M,1,0
10,21,Male,Caucasian,1,F,3,1
11,27,Male,Caucasian,0,M,4,0
12,23,Male,African-American,3,F,6,0
13,37,Male,Caucasian,0,F,1,0
14,29,Male,African-American,6,F,5,1
15,40,Female,African-American,0,M,3,0
16,28,Male,African-American,1,F,4,1
17,29,Male,Caucasian,0,F,1,0
18,30,Male,African-American,2,F,5,1
19,42,Male,Caucasian,0,F,1,0
20,30,Male,African-American,0,F,4,0
21,33,Male,African-American,3,F,6,1
22,31,Male,Caucasian,1,F,4,0
23,25,Male,African-American,2,F,7,1
24,38,Female,Caucasian,0,M,2,0
25,46,Male,African-American,11,F,8,1
26,28,Female,African-American,1,F,5,0
27,39,Male,Caucasian,0,F,1,0
28,22,Male,African-American,0,F,4,1
29,35,Male,Caucasian,2,F,3,0
30,27,Female,African-American,3,F,6,1
`;
