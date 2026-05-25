import * as React from "react";
import { PARTS, FLAT_CHAPTERS, getChapter } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";

interface PhaseTimelineProps {
  /** Highlighted chapter slug (full slug, e.g. "part1-architecture/01-system-overview"). */
  currentSlug?: string;
  /** Show chapter count under each part label. */
  variant?: "compact" | "full";
}

/**
 * Horizontal scroller showing Parts I–VIII as a progress timeline. Used both
 * on the landing page (`currentSlug` omitted, shows the whole arc) and at
 * the top of every tutorial chapter (highlights current position).
 *
 * Designed as a React island because it owns hover state, and the scroll
 * container needs to ensure the active chapter is visible — both client-side
 * concerns.
 */
export default function PhaseTimeline({
  currentSlug,
  variant = "compact",
}: PhaseTimelineProps): React.ReactElement {
  const currentIndex = currentSlug
    ? FLAT_CHAPTERS.findIndex((c) => c.fullSlug === currentSlug)
    : -1;
  const scrollerRef = React.useRef<HTMLDivElement>(null);
  const activeRef = React.useRef<HTMLAnchorElement>(null);

  React.useEffect(() => {
    const el = activeRef.current;
    const parent = scrollerRef.current;
    if (!el || !parent) return;
    // Compute the active chapter's position relative to the scroller's
    // scrollable content. Using getBoundingClientRect deltas + current
    // scrollLeft is robust to nested layouts; el.offsetLeft is not (it's
    // relative to the nearest positioned ancestor, which here is <body>).
    const elRect = el.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const elLeftInContent = elRect.left - parentRect.left + parent.scrollLeft;
    const targetLeft =
      elLeftInContent - parent.clientWidth / 2 + elRect.width / 2;
    // Clamp so we don't request a negative scroll (e.g. early chapters
    // already at the start) — the browser handles overshoot fine but a
    // negative value is wasteful and shows up in animation frames.
    parent.scrollTo({
      left: Math.max(0, targetLeft),
      behavior: "smooth",
    });
  }, [currentSlug]);

  return (
    <nav
      aria-label="Tutorial timeline"
      className="overflow-x-auto rounded-lg border bg-card"
      ref={scrollerRef}
    >
      <ol className="flex min-w-max items-stretch">
        {PARTS.map((part, partIdx) => {
          const isCurrentPart = getChapter(currentSlug ?? "")?.part.id === part.id;
          return (
            <li
              key={part.id}
              className={cn(
                "flex flex-col border-r border-border px-4 py-3 last:border-r-0",
                isCurrentPart && "bg-accent/40",
              )}
            >
              <div className="mb-2 flex items-baseline gap-2">
                <span className="text-xs font-mono text-muted-foreground">
                  {part.label}
                </span>
                <span className="text-sm font-semibold whitespace-nowrap">
                  {part.title}
                </span>
                {variant === "full" && (
                  <span className="ml-auto text-xs text-muted-foreground">
                    {part.chapters.length} ch
                  </span>
                )}
              </div>
              <ol className="flex items-center gap-1">
                {part.chapters.map((chapter) => {
                  const slug = `${part.id}/${chapter.slug}`;
                  const flatIdx = FLAT_CHAPTERS.findIndex((c) => c.fullSlug === slug);
                  const isActive = slug === currentSlug;
                  const isPast = currentIndex >= 0 && flatIdx < currentIndex;
                  return (
                    <li key={chapter.slug}>
                      <a
                        ref={isActive ? activeRef : undefined}
                        href={`/docs/${slug}`}
                        title={chapter.title}
                        aria-label={`${part.title}: ${chapter.title}`}
                        className={cn(
                          "flex h-6 min-w-6 items-center justify-center rounded-full border text-[10px] font-medium transition-colors",
                          isActive &&
                            "border-foreground bg-foreground text-background",
                          !isActive &&
                            isPast &&
                            "border-foreground/40 bg-foreground/10 text-foreground",
                          !isActive &&
                            !isPast &&
                            "border-border bg-background text-muted-foreground hover:border-foreground hover:text-foreground",
                        )}
                      >
                        {chapter.slug.slice(0, 2)}
                      </a>
                    </li>
                  );
                })}
              </ol>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
