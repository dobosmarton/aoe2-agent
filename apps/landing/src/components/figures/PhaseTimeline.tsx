import * as React from "react";
import { SECTIONS, FLAT_PAGES } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";

interface PhaseTimelineProps {
  /** Current page (full slug, e.g. "agent/overview"). */
  currentSlug: string;
}

// "+" marks inside the corners of the current section, like the detector frames on the
// rest of the site. Inset, so the scroll container does not clip them.
const CORNERS = ["top-1 left-1", "top-1 right-1", "bottom-2 left-1", "bottom-2 right-1"];

/**
 * The section strip at the top of every guide page: the five sections in one row,
 * with numbered chips for their pages. The current section gets accent corner
 * marks, the current page is the accent chip, and a thin line along the bottom
 * shows how far through the guide the page is.
 *
 * A React island because the scroll container keeps the current section in view
 * on narrow screens.
 */
export default function PhaseTimeline({ currentSlug }: PhaseTimelineProps): React.ReactElement {
  const currentIndex = FLAT_PAGES.findIndex((p) => p.fullSlug === currentSlug);
  const progress = currentIndex >= 0 ? ((currentIndex + 1) / FLAT_PAGES.length) * 100 : 0;
  const scrollerRef = React.useRef<HTMLDivElement>(null);
  const activeRef = React.useRef<HTMLLIElement>(null);

  React.useEffect(() => {
    const el = activeRef.current;
    const parent = scrollerRef.current;
    if (!el || !parent) return;
    // Centre the current section in the scroller. Bounding-rect deltas plus the
    // current scrollLeft are robust to nested layouts; offsetLeft is not.
    const elRect = el.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    const elLeftInContent = elRect.left - parentRect.left + parent.scrollLeft;
    parent.scrollTo({ left: Math.max(0, elLeftInContent - parent.clientWidth / 2 + elRect.width / 2) });
  }, [currentSlug]);

  return (
    <nav
      aria-label="Guide sections and pages"
      className="relative overflow-hidden rounded-lg border border-foreground/10 bg-card/70 backdrop-blur-sm"
    >
      {/* The edges fade out, so a strip wider than the page reads as scrollable. */}
      <div
        ref={scrollerRef}
        className="overflow-x-auto [mask-image:linear-gradient(to_right,transparent,black_1.25rem,black_calc(100%-1.25rem),transparent)] [scrollbar-width:none]"
      >
        <ol className="flex min-w-max list-none">
          {SECTIONS.map((section) => {
            const isCurrentPart = section.pages.some((p) => `${section.id}/${p.slug}` === currentSlug);
            return (
              <li
                key={section.id}
                ref={isCurrentPart ? activeRef : undefined}
                className="relative border-r border-foreground/10 px-3.5 pt-2.5 pb-3 first:pl-5 last:border-r-0 last:pr-5"
              >
                {isCurrentPart &&
                  CORNERS.map((position) => (
                    <svg
                      key={position}
                      aria-hidden
                      viewBox="0 0 7 7"
                      className={cn("absolute h-[7px] w-[7px] text-accent-foreground", position)}
                    >
                      <path d="M3.5 0v7M0 3.5h7" stroke="currentColor" strokeWidth="1" />
                    </svg>
                  ))}
                <p
                  className={cn(
                    "font-mono text-[11px] tracking-wide whitespace-nowrap uppercase",
                    isCurrentPart ? "text-accent-foreground" : "text-muted-foreground",
                  )}
                >
                  {section.title}
                </p>
                <ol className="mt-1.5 flex list-none items-center gap-0.5">
                  {section.pages.map((page, pageIndex) => {
                    const slug = `${section.id}/${page.slug}`;
                    const index = FLAT_PAGES.findIndex((p) => p.fullSlug === slug);
                    const isActive = slug === currentSlug;
                    const isPast = currentIndex >= 0 && index < currentIndex;
                    return (
                      <li key={page.slug}>
                        <a
                          href={`/docs/${slug}`}
                          title={page.title}
                          aria-label={`${section.title}: ${page.title}`}
                          aria-current={isActive ? "page" : undefined}
                          className={cn(
                            "block rounded px-1.5 py-0.5 font-mono text-xs tabular-nums transition-colors",
                            isActive && "bg-accent-foreground/15 text-accent-foreground",
                            !isActive && isPast && "text-foreground hover:text-accent-foreground",
                            !isActive && !isPast && "text-muted-foreground hover:text-accent-foreground",
                          )}
                        >
                          {String(pageIndex + 1).padStart(2, "0")}
                        </a>
                      </li>
                    );
                  })}
                </ol>
              </li>
            );
          })}
        </ol>
      </div>
      {/* How far through the guide this page is. */}
      <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5">
        <span className="block h-0.5 bg-accent-foreground" style={{ width: `${progress}%` }} />
      </span>
    </nav>
  );
}
