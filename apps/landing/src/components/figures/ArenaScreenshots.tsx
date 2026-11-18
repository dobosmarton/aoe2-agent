import * as React from "react";
import { Dialog, Heading, Modal, ModalOverlay } from "react-aria-components";
import { ExternalLink, X } from "lucide-react";

import { cn } from "@/lib/utils";

type Shot = {
  /** Image filename under /screenshots/arena/. Provide a real PNG before launch. */
  src: string;
  panel: string;
  /** Short description shown under the thumbnail. */
  caption: string;
  /** Doc the panel is described in. */
  docHref: string;
  docLabel: string;
};

const SHOTS: Shot[] = [
  {
    src: "/screenshots/arena/world.png",
    panel: "World",
    caption: "Resources, population against cap, age and building counts across the turns of a run.",
    docHref: "/docs/arena/world-sim",
    docLabel: "World sim",
  },
  {
    src: "/screenshots/arena/trace.png",
    panel: "Trace",
    caption: "Every event of the run, grouped by turn, each expandable to its full JSON payload.",
    docHref: "/docs/arena/event-log-and-replay",
    docLabel: "Event log and replay",
  },
  {
    src: "/screenshots/arena/diff.png",
    panel: "Diff",
    caption: "A fork's lineage and its state next to the parent run's state at the fork turn.",
    docHref: "/docs/web/run-viewer",
    docLabel: "Run viewer",
  },
  {
    src: "/screenshots/arena/operator.png",
    panel: "Operator",
    caption: "Fork a run from any turn, optionally patching resources, population or age, and replay it.",
    docHref: "/docs/web/run-viewer",
    docLabel: "Run viewer",
  },
];

/**
 * Lightbox grid for arena UI screenshots. Each thumbnail opens a modal
 * dialog with the full-size image and caption.
 *
 * Run `just capture-screenshots` (or `bun --filter aoe2-llm-arena-web
 * capture:screenshots`) to populate the PNGs from a live dashboard via
 * Playwright. Source: `apps/landing/scripts/capture-arena-screenshots.ts`.
 */
export const ArenaScreenshots = (): React.ReactElement => {
  const [openIndex, setOpenIndex] = React.useState<number | null>(null);
  const open = openIndex === null ? null : SHOTS[openIndex];

  return (
    // Radix needed the grid inside Dialog.Root; react-aria's ModalOverlay is
    // self-contained, so the trigger grid and the modal are plain siblings.
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {SHOTS.map((shot, i) => (
          <button
            key={shot.panel}
            type="button"
            onClick={() => setOpenIndex(i)}
            className="group rounded-lg border border-foreground/10 bg-card p-3 text-left transition-[border-color,box-shadow] duration-200 hover:border-accent-foreground/50 hover:shadow-[0_12px_32px_-16px_rgb(0_0_0/0.25)]"
          >
            <ScreenshotPlaceholder src={shot.src} panel={shot.panel} />
            <p className="mt-3 font-heading font-bold group-hover:text-accent-foreground">{shot.panel}</p>
            <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
              {shot.caption}
            </p>
            <p className="mt-2 inline-flex items-center gap-1 font-mono text-[11px] tracking-wider text-muted-foreground uppercase">
              {shot.docLabel}
              <ExternalLink className="h-3 w-3" />
            </p>
          </button>
        ))}
      </div>

      <ModalOverlay
        isOpen={open !== null}
        onOpenChange={(isOpen) => {
          if (!isOpen) {
            setOpenIndex(null);
          }
        }}
        isDismissable
        className="fixed inset-0 z-50 bg-black/70"
      >
        <Modal className="fixed left-1/2 top-1/2 z-50 w-[min(90vw,1100px)] max-h-[90vh] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border bg-popover p-4 shadow-xl">
          <Dialog aria-describedby={undefined} className="outline-none">
          {open && (
            <div>
              <div className="mb-3 flex items-start justify-between gap-4">
                <div>
                  <Heading slot="title" className="text-lg font-semibold">
                    {open.panel} panel
                  </Heading>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {open.caption}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setOpenIndex(null);
                  }}
                  className="rounded-md p-1 hover:bg-accent"
                  aria-label="Close"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <ScreenshotPlaceholder
                src={open.src}
                panel={open.panel}
                large
              />
              <a
                href={open.docHref}
                className="mt-3 inline-flex items-center gap-1 text-sm underline"
              >
                Read about the {open.docLabel} panel
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          )}
          </Dialog>
        </Modal>
      </ModalOverlay>
    </>
  );
};

/**
 * If the screenshot PNG exists it renders normally; if not, we show a
 * stylized placeholder so the landing page looks complete even before
 * `capture-arena-screenshots.ts` has been run.
 */
type ScreenshotPlaceholderProps = {
  src: string;
  panel: string;
  large?: boolean;
};

const ScreenshotPlaceholder = ({ src, panel, large = false }: ScreenshotPlaceholderProps): React.ReactElement => {
  const [errored, setErrored] = React.useState(false);

  if (errored) {
    return (
      <div
        className="flex aspect-video w-full items-center justify-center rounded-md border border-dashed bg-muted text-xs text-muted-foreground"
      >
        <span className="font-mono">{panel}.png · not captured yet</span>
      </div>
    );
  }

  return (
    <img
      src={src}
      alt={`${panel} panel of the arena UI`}
      className={cn("w-full rounded-md border border-foreground/10", large && "rounded-lg")}
      loading="lazy"
      onError={() => setErrored(true)}
    />
  );
};
