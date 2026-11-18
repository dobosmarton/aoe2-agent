import * as React from "react";
import { SECTIONS, FLAT_PAGES, type Section } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";
import { Corners } from "@/components/ui/frame";

type PhaseTimelineProps = {
  /** Current page (full slug, e.g. "agent/overview"). */
  currentSlug: string;
};

// "+" marks inside the corners of the current section, like the detector frames on the
// rest of the site. Inset, so the scroll container does not clip them.
const SECTION_CORNERS = ["top-1 left-1", "top-1 right-1", "bottom-2 left-1", "bottom-2 right-1"];

const pageIndexOf = (fullSlug: string): number => FLAT_PAGES.findIndex((page) => page.fullSlug === fullSlug);

const chipStateClass = (isActive: boolean, isPast: boolean): string => {
  if (isActive) return "bg-accent-foreground/15 text-accent-foreground";
  if (isPast) return "text-foreground hover:text-accent-foreground";
  return "text-muted-foreground hover:text-accent-foreground";
};

// Centres the current section in the scroller. Bounding-rect deltas plus the current
// scrollLeft are robust to nested layouts; offsetLeft is not.
const centreInScroller = (item: HTMLElement, scroller: HTMLElement): void => {
  const itemRect = item.getBoundingClientRect();
  const scrollerRect = scroller.getBoundingClientRect();
  const itemLeftInContent = itemRect.left - scrollerRect.left + scroller.scrollLeft;
  scroller.scrollTo({ left: Math.max(0, itemLeftInContent - scroller.clientWidth / 2 + itemRect.width / 2) });
};

type SectionItemProps = {
  section: Section;
  currentSlug: string;
  currentIndex: number;
  activeRef: React.RefObject<HTMLLIElement | null>;
};

const SectionItem = ({ section, currentSlug, currentIndex, activeRef }: SectionItemProps): React.ReactElement => {
  const isCurrentPart = section.pages.some((page) => `${section.id}/${page.slug}` === currentSlug);
  return (
    <li
      ref={isCurrentPart ? activeRef : undefined}
      className="relative border-r border-foreground/10 px-3.5 pt-2.5 pb-3 first:pl-5 last:border-r-0 last:pr-5"
    >
      {isCurrentPart && <Corners positions={SECTION_CORNERS} className="text-accent-foreground" />}
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
          const isActive = slug === currentSlug;
          const isPast = currentIndex >= 0 && pageIndexOf(slug) < currentIndex;
          return (
            <li key={page.slug}>
              <a
                href={`/docs/${slug}`}
                title={page.title}
                aria-label={`${section.title}: ${page.title}`}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "block rounded px-1.5 py-0.5 font-mono text-xs tabular-nums transition-colors",
                  chipStateClass(isActive, isPast),
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
};

/**
 * The section strip at the top of every guide page: the five sections in one row,
 * with numbered chips for their pages. The current section gets accent corner
 * marks, the current page is the accent chip, and a thin line along the bottom
 * shows how far through the guide the page is.
 *
 * A React island because the scroll container keeps the current section in view
 * on narrow screens.
 */
export const PhaseTimeline = ({ currentSlug }: PhaseTimelineProps): React.ReactElement => {
  const currentIndex = pageIndexOf(currentSlug);
  const progress = currentIndex >= 0 ? ((currentIndex + 1) / FLAT_PAGES.length) * 100 : 0;
  const scrollerRef = React.useRef<HTMLDivElement>(null);
  const activeRef = React.useRef<HTMLLIElement>(null);

  React.useEffect(() => {
    const item = activeRef.current;
    const scroller = scrollerRef.current;
    if (item && scroller) centreInScroller(item, scroller);
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
          {SECTIONS.map((section) => (
            <SectionItem
              key={section.id}
              section={section}
              currentSlug={currentSlug}
              currentIndex={currentIndex}
              activeRef={activeRef}
            />
          ))}
        </ol>
      </div>
      {/* How far through the guide this page is. */}
      <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5">
        <span className="block h-0.5 bg-accent-foreground" style={{ width: `${progress}%` }} />
      </span>
    </nav>
  );
};
