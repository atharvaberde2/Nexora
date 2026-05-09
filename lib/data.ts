// Mock audit result — modeled after the COMPAS narrative in the project brief.
// Numbers are illustrative but internally consistent (Pareto frontier, dominance,
// statistical significance) so the demo holds up under scrutiny.

export type ModelKey = "logistic" | "rf" | "xgb" | "lgbm" | "svm";

export type Model = {
  key: ModelKey;
  name: string;
  family: string;
  color: string;            // tailwind token name (m1..m5)
  auc: number;              // overall AUC
  aucCi: [number, number];
  eqOddsGap: number;        // 0 = perfectly fair
  eqOddsCi: [number, number];
  ellipse: { rx: number; ry: number; rot: number }; // bootstrap confidence shape
  paretoOptimal: boolean;
  recommended?: boolean;
  dpGap: number;
  ece: number;
  fprWhite: number;
  fprBlack: number;
  delongP: number;
};

export const MODELS: Model[] = [
  {
    key: "logistic",
    name: "Logistic Regression",
    family: "Linear · interpretable",
    color: "m1",
    auc: 0.872,
    aucCi: [0.861, 0.883],
    eqOddsGap: 0.041,
    eqOddsCi: [0.029, 0.054],
    ellipse: { rx: 14, ry: 9, rot: -8 },
    paretoOptimal: true,
    recommended: true,
    dpGap: 0.062,
    ece: 0.018,
    fprWhite: 0.214,
    fprBlack: 0.255,
    delongP: 0.211,
  },
  {
    key: "lgbm",
    name: "LightGBM",
    family: "Gradient boosting",
    color: "m4",
    auc: 0.881,
    aucCi: [0.870, 0.892],
    eqOddsGap: 0.069,
    eqOddsCi: [0.054, 0.084],
    ellipse: { rx: 11, ry: 7, rot: 6 },
    paretoOptimal: true,
    dpGap: 0.084,
    ece: 0.022,
    fprWhite: 0.198,
    fprBlack: 0.267,
    delongP: 0.041,
  },
  {
    key: "rf",
    name: "Random Forest",
    family: "Ensemble · bagged trees",
    color: "m2",
    auc: 0.864,
    aucCi: [0.852, 0.876],
    eqOddsGap: 0.082,
    eqOddsCi: [0.067, 0.097],
    ellipse: { rx: 10, ry: 8, rot: 12 },
    paretoOptimal: false,
    dpGap: 0.108,
    ece: 0.031,
    fprWhite: 0.221,
    fprBlack: 0.303,
    delongP: 0.018,
  },
  {
    key: "xgb",
    name: "XGBoost",
    family: "Gradient boosting",
    color: "m3",
    auc: 0.890,
    aucCi: [0.879, 0.901],
    eqOddsGap: 0.118,
    eqOddsCi: [0.103, 0.134],
    ellipse: { rx: 13, ry: 9, rot: 18 },
    paretoOptimal: true,
    dpGap: 0.142,
    ece: 0.027,
    fprWhite: 0.193,
    fprBlack: 0.311,
    delongP: 0.007,
  },
  {
    key: "svm",
    name: "Linear SVM",
    family: "Max-margin classifier",
    color: "m5",
    auc: 0.851,
    aucCi: [0.838, 0.864],
    eqOddsGap: 0.094,
    eqOddsCi: [0.078, 0.110],
    ellipse: { rx: 12, ry: 9, rot: -2 },
    paretoOptimal: false,
    dpGap: 0.121,
    ece: 0.044,
    fprWhite: 0.228,
    fprBlack: 0.322,
    delongP: 0.029,
  },
];

export const ROOT_CAUSE = [
  { label: "Proxy discrimination", value: 0.73, color: "danger" },
  { label: "Threshold effect", value: 0.14, color: "warning" },
  { label: "Calibration drift", value: 0.07, color: "ink-muted" },
  { label: "Sample imbalance", value: 0.04, color: "ink-muted" },
  { label: "Label bias", value: 0.02, color: "ink-muted" },
];

export const DATASET = {
  name: "COMPAS · Recidivism",
  records: 5278,
  features: 12,
  protectedAttr: "race",
  positiveRate: 0.466,
  groups: [
    { name: "African-American", n: 3175, baseRate: 0.519 },
    { name: "Caucasian", n: 2103, baseRate: 0.394 },
  ],
};

export const POPULATION_IMPACT = 1847;

export const REMEDIATION = [
  {
    status: "blocked" as const,
    title: "Threshold adjustment",
    body: "Root cause is proxy discrimination. Group-specific thresholds would mask, not fix, the underlying mechanism. Blocked by the remediation engine.",
  },
  {
    status: "recommended" as const,
    title: "Decorrelate priors_count from race",
    body: "SHAP rank delta: priors_count is the #1 driver for African-American defendants in XGBoost (#4 in Logistic). Permutation test p=0.003.",
  },
  {
    status: "optional" as const,
    title: "Group-calibrated probabilities",
    body: "ECE drops 0.027 → 0.011 for African-American group after Platt scaling. Bootstrap-validated as a real Pareto improvement.",
  },
];
