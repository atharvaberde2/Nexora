# Nexora

> Five models enter. The fair one ships.

Pre-deployment fairness audit UI — Next.js 14 + Tailwind, dark-first, editorial-scientific design.

## Run

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Routes

- `/` — landing page
- `/audit` — the demo audit (the showpiece — Pareto frontier, leaderboard, root cause, three-audience reports)

## Design system

- Type: Inter (UI) · Instrument Serif (display) · JetBrains Mono (metrics)
- Color: dark canvas `#08090B`, single accent `#5B7FFF` reserved for the recommended model and primary CTAs
- Spacing: 4-based scale, defined in `tailwind.config.ts`
- All design tokens live in `tailwind.config.ts` and `app/globals.css`

## Structure

```
app/
  layout.tsx       fonts + metadata
  globals.css      base styles, grid background, scrollbar, focus rings
  page.tsx         landing
  audit/page.tsx   audit result — the demo
components/
  pareto-chart.tsx custom SVG Pareto chart with bootstrap ellipses
  leaderboard.tsx  sortable model leaderboard
  report-tabs.tsx  Engineer / Regulator / Community report views
  primitives.tsx   Logo, Button, Card, Badge, Stat
  nav.tsx          top nav
lib/
  data.ts          mock COMPAS-style audit data
  cn.ts            tailwind merge helper
```
