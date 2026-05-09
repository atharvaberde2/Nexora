import Link from "next/link";
import { Logo, Button } from "./primitives";

export function Nav({
  context,
  status,
}: {
  context?: { kind: string; name: string };
  status?: string;
}) {
  return (
    <header className="sticky top-0 z-30 backdrop-blur-md bg-canvas/80 border-b border-hairline">
      <div className="max-w-[1320px] mx-auto h-14 px-5 sm:px-8 flex items-center gap-4">
        <Link href="/" className="flex items-center" aria-label="Nexora home">
          <Logo />
        </Link>

        {context && (
          <>
            <span className="text-ink-faint">/</span>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-ink-dim font-mono text-xs uppercase tracking-wider">
                {context.kind}
              </span>
              <span className="text-ink-muted">·</span>
              <span className="text-ink">{context.name}</span>
            </div>
          </>
        )}

        <div className="ml-auto flex items-center gap-2">
          {status && (
            <div className="hidden sm:flex items-center gap-2 mr-2">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-75 animate-pulse-ring" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-success" />
              </span>
              <span className="text-2xs text-ink-muted font-mono tracking-wider uppercase">
                {status}
              </span>
            </div>
          )}
          <Button variant="ghost" size="sm" className="hidden md:inline-flex">
            Docs
          </Button>
          <kbd className="hidden md:inline-flex h-7 items-center gap-1 rounded border border-line bg-elevated px-1.5 text-2xs font-mono text-ink-dim">
            ⌘K
          </kbd>
          <Button variant="secondary" size="sm">
            Share
          </Button>
        </div>
      </div>
    </header>
  );
}
