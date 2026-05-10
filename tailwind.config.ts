import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Softened palette: subtle warming on the dark surfaces and gentler
        // semantic colors to reduce the "robotic console" feel without losing
        // the clean professional dark theme. Contrast ratios still ≥ 4.5:1
        // for text on canvas (WCAG AA).
        canvas: "#0C0E13",
        surface: "#14161C",
        elevated: "#1A1D25",
        hairline: "#22262F",
        line: "#2D323C",
        ink: {
          DEFAULT: "#EAECEF",
          muted: "#A8AEB8",
          dim: "#737A85",
          faint: "#444A55",
        },
        accent: {
          DEFAULT: "#738BF2",
          soft: "rgba(115, 139, 242, 0.14)",
          ring: "rgba(115, 139, 242, 0.32)",
        },
        success: "#5DC78A",
        warning: "#E5B95A",
        danger: "#E68585",
        // Model series — softened to match palette
        m1: "#A4ABB6", // logistic
        m2: "#B4A0F5", // random forest
        m3: "#E68585", // xgboost
        m4: "#5BCFA1", // lightgbm
        m5: "#E5B95A", // svm
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        serif: ["var(--font-instrument)", "Georgia", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["11px", { lineHeight: "14px", letterSpacing: "0.04em" }],
        xs: ["12px", { lineHeight: "16px" }],
        sm: ["13px", { lineHeight: "20px" }],
        base: ["14px", { lineHeight: "22px" }],
        md: ["16px", { lineHeight: "26px" }],
        lg: ["20px", { lineHeight: "28px" }],
        xl: ["28px", { lineHeight: "34px", letterSpacing: "-0.01em" }],
        "2xl": ["40px", { lineHeight: "46px", letterSpacing: "-0.02em" }],
        "3xl": ["56px", { lineHeight: "60px", letterSpacing: "-0.025em" }],
        display: ["88px", { lineHeight: "0.95", letterSpacing: "-0.035em" }],
      },
      spacing: {
        "0.5": "2px",
        "1.5": "6px",
        "2.5": "10px",
        "3.5": "14px",
        "18": "72px",
        "22": "88px",
      },
      borderRadius: {
        sm: "4px",
        DEFAULT: "6px",
        md: "8px",
        lg: "12px",
        xl: "16px",
        "2xl": "20px",
      },
      boxShadow: {
        "panel": "0 1px 0 rgba(255,255,255,0.04) inset, 0 0 0 1px #22262F",
        "lift": "0 1px 0 rgba(255,255,255,0.06) inset, 0 0 0 1px #2D323C, 0 12px 40px -16px rgba(0,0,0,0.55)",
        "glow": "0 0 0 1px rgba(115,139,242,0.4), 0 0 32px -4px rgba(115,139,242,0.4)",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(0.96)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        "draw": {
          from: { strokeDashoffset: "1000" },
          to: { strokeDashoffset: "0" },
        },
        "pulse-ring": {
          "0%, 100%": { transform: "scale(1)", opacity: "0.5" },
          "50%": { transform: "scale(1.4)", opacity: "0" },
        },
      },
      animation: {
        "fade-up": "fade-up 600ms cubic-bezier(0.22, 1, 0.36, 1) both",
        "fade-in": "fade-in 500ms ease-out both",
        "scale-in": "scale-in 500ms cubic-bezier(0.22, 1, 0.36, 1) both",
        "draw": "draw 1.4s cubic-bezier(0.65, 0, 0.35, 1) both",
        "pulse-ring": "pulse-ring 2.4s ease-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
