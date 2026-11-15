import * as React from "react";
import { SECTIONS, FLAT_PAGES, getPage } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";

interface PhaseTimelineProps {
  /** Highlighted page slug (full slug, e.g. "agent/overview"). */
  currentSlug?: string;
  /** Show the page count next to each section title. */
  variant?: "compact" | "full";
}

/**
 * Horizontal scroller showing the documentation sections as a progress
 * timeline. Used both on the landing page (`currentSlug` omitted, shows the
 * whole arc) and at the top of every section page (highlights the current
 * position).
 *
 * Designed as a React island because it owns hover state, and the scroll
 * container needs to keep the active page visible. Both are client-side
 * concerns.
 */
export default function PhaseTimeline({
  currentSlug,
  variant = "compact",
}: PhaseTimelineProps): React.ReactElement {
  const currentIndex = currentSlug
    ? FLAT_PAGES.findIndex((p) => p.fullSlug === currentSlug)
    : -1;
  const scrollerRef = React.useRef<HTMLDivElement>(null);
  const activeRef = React.useRef<HTMLAnchorElement>(null);

  React.useEffect(() => {
    const el = activeRef.current;
    const parent = scrollerRef.current;
    if (!el || !parent) return;
    // Compute the active page's position relative to the scroller's
    // scrollable content. Using getBoundingClientRect deltas + current
    // scrollLeft is robust to nested layouts; el.offsetLeft is not (it's
    // relative to the nearest positioned ancestor, which here is <body>).
    const elRect = el.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const elLeftInContent = elRect.left - parentRect.left + parent.scrollLeft;
    const targetLeft =
      elLeftInContent - parent.clientWidth / 2 + elRect.width / 2;
    // Clamp so we don't request a negative scroll (e.g. early pages
    // already at the start). The browser handles overshoot fine, but a
    // negative value is wasteful and shows up in animation frames.
    parent.scrollTo({
      left: Math.max(0, targetLeft),
      behavior: "smooth",
    });
  }, [currentSlug]);

  return (
    <nav
      aria-label="Documentation timeline"
      className="overflow-x-auto rounded-lg border bg-card"
      ref={scrollerRef}
    >
      <ol className="flex min-w-max items-stretch">
        {SECTIONS.map((section) => {
          const isCurrentSection = getPage(currentSlug ?? "")?.section.id === section.id;
          return (
            <li
              key={section.id}
              className={cn(
                "flex flex-col border-r border-border px-4 py-3 last:border-r-0",
                isCurrentSection && "bg-accent/40",
              )}
            >
              <div className="mb-2 flex items-baseline gap-2">
                <span className="text-sm font-semibold whitespace-nowrap">
                  {section.title}
                </span>
                {variant === "full" && (
                  <span className="ml-auto text-xs text-muted-foreground">
                    {section.pages.length} pages
                  </span>
                )}
              </div>
              <ol className="flex items-center gap-1">
                {section.pages.map((page, pageIdx) => {
                  const slug = `${section.id}/${page.slug}`;
                  const flatIdx = FLAT_PAGES.findIndex((p) => p.fullSlug === slug);
                  const isActive = slug === currentSlug;
                  const isPast = currentIndex >= 0 && flatIdx < currentIndex;
                  return (
                    <li key={page.slug}>
                      <a
                        ref={isActive ? activeRef : undefined}
                        href={`/docs/${slug}`}
                        title={page.title}
                        aria-label={`${section.title}: ${page.title}`}
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
                        {pageIdx + 1}
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
