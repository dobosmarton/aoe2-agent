import * as React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { ExternalLink, X } from "lucide-react";

import { cn } from "@/lib/utils";

interface Shot {
  /** Image filename under /screenshots/arena/. Provide a real PNG before launch. */
  src: string;
  panel: string;
  /** Short description shown under the thumbnail. */
  caption: string;
  /** Doc the panel is described in. */
  docHref: string;
  docLabel: string;
}

const SHOTS: Shot[] = [
  {
    src: "/screenshots/arena/world.png",
    panel: "World",
    caption: "Rendered WorldState projection over time — resources, units, buildings as the executor sees them.",
    docHref: "/docs/part6-evaluation-arena/18-synthetic-world-sim",
    docLabel: "Synthetic World Sim",
  },
  {
    src: "/screenshots/arena/trace.png",
    panel: "Trace",
    caption: "Per-turn LLM trace: prompt, response, parsed action list, latency.",
    docHref: "/docs/part6-evaluation-arena/16-duckdb-persister-and-replay",
    docLabel: "DuckDB Persister & Replay",
  },
  {
    src: "/screenshots/arena/diff.png",
    panel: "Diff",
    caption: "Side-by-side comparison of two forks of the same run — what diverged turn-by-turn.",
    docHref: "/docs/part7-arena-web/20-fork-and-diff-ui",
    docLabel: "Fork and Diff UI",
  },
  {
    src: "/screenshots/arena/operator.png",
    panel: "Operator",
    caption: "Fork-from-turn primitive: replay any run from any point with a different prompt or model.",
    docHref: "/docs/part7-arena-web/20-fork-and-diff-ui",
    docLabel: "Fork and Diff UI",
  },
];

/**
 * Lightbox grid for arena UI screenshots. Each thumbnail opens a Radix
 * Dialog with the full-size image and caption.
 *
 * Run `just capture-screenshots` (or `bun --filter aoe2-llm-arena-web
 * capture:screenshots`) to populate the PNGs from a live dashboard via
 * Playwright. Source: `apps/landing/scripts/capture-arena-screenshots.ts`.
 */
export default function ArenaScreenshots(): React.ReactElement {
  const [openIdx, setOpenIdx] = React.useState<number | null>(null);
  return (
    <Dialog.Root open={openIdx !== null} onOpenChange={(v) => !v && setOpenIdx(null)}>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {SHOTS.map((shot, i) => (
          <button
            key={shot.panel}
            type="button"
            onClick={() => setOpenIdx(i)}
            className="group rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground"
          >
            <ScreenshotPlaceholder src={shot.src} panel={shot.panel} />
            <p className="mt-3 text-sm font-semibold">{shot.panel}</p>
            <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
              {shot.caption}
            </p>
            <p className="mt-2 inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              {shot.docLabel}
              <ExternalLink className="h-3 w-3" />
            </p>
          </button>
        ))}
      </div>

      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-1/2 z-50 w-[min(90vw,1100px)] max-h-[90vh] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border bg-popover p-4 shadow-xl"
        >
          {openIdx !== null && (
            <div>
              <div className="mb-3 flex items-start justify-between gap-4">
                <div>
                  <Dialog.Title className="text-lg font-semibold">
                    {SHOTS[openIdx]!.panel} panel
                  </Dialog.Title>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {SHOTS[openIdx]!.caption}
                  </p>
                </div>
                <Dialog.Close className="rounded-md p-1 hover:bg-accent" aria-label="Close">
                  <X className="h-4 w-4" />
                </Dialog.Close>
              </div>
              <ScreenshotPlaceholder
                src={SHOTS[openIdx]!.src}
                panel={SHOTS[openIdx]!.panel}
                large
              />
              <a
                href={SHOTS[openIdx]!.docHref}
                className="mt-3 inline-flex items-center gap-1 text-sm underline"
              >
                Read about the {SHOTS[openIdx]!.docLabel} panel
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/**
 * If the screenshot PNG exists it renders normally; if not, we show a
 * stylized placeholder so the landing page looks complete even before
 * `capture-arena-screenshots.ts` has been run.
 */
function ScreenshotPlaceholder({
  src,
  panel,
  large = false,
}: {
  src: string;
  panel: string;
  large?: boolean;
}): React.ReactElement {
  const [errored, setErrored] = React.useState(false);
  if (errored) {
    return (
      <div
        className={cn(
          "flex w-full items-center justify-center rounded-md border border-dashed bg-muted text-xs text-muted-foreground",
          large ? "aspect-video" : "aspect-video",
        )}
      >
        <span className="font-mono">{panel}.png · not captured yet</span>
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={`${panel} panel of the arena UI`}
      className={cn("w-full rounded-md border", large && "rounded-lg")}
      loading="lazy"
      onError={() => setErrored(true)}
    />
  );
}
