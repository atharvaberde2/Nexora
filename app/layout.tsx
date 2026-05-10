import type { Metadata } from "next";
import { Plus_Jakarta_Sans, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Plus Jakarta Sans — modern, slightly warmer than Inter, very readable.
// Same `--font-inter` CSS variable name kept for backward compatibility with
// the existing Tailwind fontFamily mapping; only the underlying typeface changes.
const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-inter",
  display: "swap",
});

const instrument = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-instrument",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Nexora — Five models enter. The fair one ships.",
  description:
    "Pre-deployment fairness audit. Train five model families on your data and prove — statistically — which one is fair to deploy, before a single real person is affected.",
  metadataBase: new URL("https://nexora.ai"),
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${instrument.variable} ${mono.variable}`}
    >
      <body className="bg-canvas text-ink font-sans antialiased">
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
