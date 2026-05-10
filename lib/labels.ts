// Human-readable labels for common categorical encodings.
//
// Mandate: never display a raw encoded value (sex=1, fbs=0, Loan_Status=Y)
// without translating it into something a non-technical reader can understand.
// Every stage that talks about a group should pipe through `formatGroup`.

export type LabelFormat = {
  /** The raw value as stored in the CSV (e.g., "1", "0", "Y") */
  raw: string;
  /** Short human-readable label (e.g., "Male", "Loan Approved"); null if unknown */
  display: string | null;
  /** Combined "Male (sex = 1)" form for prose */
  full: string;
  /** Same combined form, but uses the raw value alone if no display known */
  withAttr: string;
  /** True if we resolved a known mapping for this attribute */
  known: boolean;
};

/** Per-attribute → per-raw-value display label. Values lowercased on lookup. */
const KNOWN_MAPPINGS: Record<string, Record<string, string>> = {
  sex: {
    "0": "Female",
    "1": "Male",
    f: "Female",
    m: "Male",
    female: "Female",
    male: "Male",
  },
  gender: {
    "0": "Female",
    "1": "Male",
    f: "Female",
    m: "Male",
    female: "Female",
    male: "Male",
  },
  fbs: {
    "0": "Normal fasting blood sugar (≤120 mg/dL)",
    "1": "High fasting blood sugar (>120 mg/dL)",
  },
  exang: {
    "0": "No exercise-induced angina",
    "1": "Exercise-induced angina",
  },
  cp: {
    "0": "Typical angina",
    "1": "Atypical angina",
    "2": "Non-anginal pain",
    "3": "Asymptomatic",
  },
  restecg: {
    "0": "Normal ECG",
    "1": "ST-T abnormality",
    "2": "Left ventricular hypertrophy",
  },
  thal: {
    "1": "Normal",
    "2": "Fixed defect",
    "3": "Reversible defect",
  },
  slope: {
    "0": "Upsloping ST",
    "1": "Flat ST",
    "2": "Downsloping ST",
  },
  loan_status: {
    y: "Loan Approved",
    n: "Loan Denied",
    "1": "Loan Approved",
    "0": "Loan Denied",
  },
  married: {
    y: "Married",
    n: "Not married",
    "1": "Married",
    "0": "Not married",
  },
  self_employed: {
    y: "Self-employed",
    n: "Salaried",
    "1": "Self-employed",
    "0": "Salaried",
  },
  education: {
    graduate: "Graduate",
    "not graduate": "Not Graduate",
  },
  property_area: {
    urban: "Urban area",
    rural: "Rural area",
    semiurban: "Semi-urban area",
  },
  race: {
    "african-american": "African-American",
    caucasian: "Caucasian",
    hispanic: "Hispanic",
    asian: "Asian",
    other: "Other race",
  },
  target: {
    "0": "Negative class",
    "1": "Positive class",
  },
};

/** Optional: short labels used inside narrow table cells where the full
 *  description doesn't fit. Falls back to the long display when missing. */
const SHORT_DISPLAY: Record<string, Record<string, string>> = {
  fbs: { "0": "Normal blood sugar", "1": "High blood sugar" },
  exang: { "0": "No angina", "1": "Angina" },
};

/** Plain-English noun-phrase for the protected attribute itself, used in prose
 *  like "underrepresented in <fasting blood sugar>". Falls back to the raw name. */
const ATTR_LABEL: Record<string, string> = {
  sex: "sex",
  gender: "gender",
  fbs: "fasting blood sugar",
  exang: "exercise-induced angina",
  cp: "chest pain type",
  restecg: "resting ECG",
  thal: "thalassemia type",
  slope: "ST slope",
  race: "race",
  loan_status: "loan status",
  married: "marital status",
  self_employed: "employment type",
  education: "education level",
  property_area: "property area",
};

export function formatGroup(attr: string, raw: unknown): LabelFormat {
  const rawStr = String(raw ?? "").trim();
  const key = (attr ?? "").toLowerCase().trim();
  const valueLc = rawStr.toLowerCase();
  const map = KNOWN_MAPPINGS[key];
  const display = map?.[valueLc] ?? null;
  if (display) {
    return {
      raw: rawStr,
      display,
      full: `${display} (${attr} = ${rawStr})`,
      withAttr: `${display} (${attr} = ${rawStr})`,
      known: true,
    };
  }
  return {
    raw: rawStr,
    display: null,
    full: rawStr,
    withAttr: `${attr} = ${rawStr}`,
    known: false,
  };
}

/** Use a shorter display if available (for table cells where space is tight). */
export function shortLabel(attr: string, raw: unknown): string {
  const key = (attr ?? "").toLowerCase().trim();
  const rawStr = String(raw ?? "").trim();
  const valueLc = rawStr.toLowerCase();
  const short = SHORT_DISPLAY[key]?.[valueLc];
  if (short) return short;
  const fmt = formatGroup(attr, raw);
  return fmt.display ?? fmt.raw;
}

/** Plain-English noun for the protected attribute itself. */
export function attrLabel(attr: string): string {
  return ATTR_LABEL[(attr ?? "").toLowerCase().trim()] ?? attr;
}

/** Returns true if no known mapping exists, so callers can render a fallback. */
export function isUnknownEncoding(attr: string, raw: unknown): boolean {
  return !formatGroup(attr, raw).known;
}

export const UNKNOWN_MAPPING_HINT =
  "(category label not provided — add metadata for interpretability)";
