import * as React from "react";

import { cn } from "@/lib/utils";

// The site's shared visual language, the same as the author's blog: hairline cells
// with header and footer strips, mono [ BRACKET ] labels, "+" registration marks, and
// one accent (the link blue). These render to static HTML in Astro pages; none of
// them needs a client directive.

const monoLabel = "font-mono text-[11px] tracking-wider text-muted-foreground uppercase";

export const cellClass = "overflow-hidden rounded-lg border border-foreground/10 bg-card";
export const cellHoverClass = "transition-shadow duration-300 hover:shadow-[0_12px_32px_-16px_rgb(0_0_0/0.25)]";

type BracketLabelProps = { children: React.ReactNode; className?: string };

export const BracketLabel = ({ children, className }: BracketLabelProps): React.ReactElement => (
  <span className={cn(monoLabel, className)}>
    <span aria-hidden>[ </span>
    {children}
    <span aria-hidden> ]</span>
  </span>
);

type SectionLabelProps = {
  children: React.ReactNode;
  aside?: React.ReactNode;
  as?: "h2" | "h3" | "p" | "div";
  id?: string;
  className?: string;
};

// A bracket label followed by a hairline rule, with an optional note at the end.
export const SectionLabel = ({ children, aside, as: Tag = "h2", id, className }: SectionLabelProps): React.ReactElement => (
  <Tag id={id} className={cn("flex items-center gap-3", className)}>
    <BracketLabel>{children}</BracketLabel>
    <span aria-hidden className="h-px flex-1 bg-foreground/10" />
    {aside && <span className={monoLabel}>{aside}</span>}
  </Tag>
);

type CellHeaderProps = {
  label: React.ReactNode;
  aside?: React.ReactNode;
  className?: string;
};

export const CellHeader = ({ label, aside, className }: CellHeaderProps): React.ReactElement => (
  <div className={cn("flex items-center justify-between gap-3 border-b border-foreground/10 px-4 py-2.5", className)}>
    <BracketLabel>{label}</BracketLabel>
    {aside && <span className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">{aside}</span>}
  </div>
);

type CellFooterProps = { children: React.ReactNode; className?: string };

export const CellFooter = ({ children, className }: CellFooterProps): React.ReactElement => (
  <div
    className={cn(
      "border-t border-foreground/10 px-4 py-2.5 font-mono text-[11px] leading-snug text-muted-foreground",
      className,
    )}
  >
    {children}
  </div>
);

const CORNER_POSITIONS = ["-top-1 -left-1", "-top-1 -right-1", "-bottom-1 -left-1", "-bottom-1 -right-1"];

type CornersProps = {
  className?: string;
  // Tailwind placement for each mark; by default they sit just outside the corners.
  positions?: readonly string[];
};

// "+" marks on each corner of the nearest positioned parent, like intersections on a
// layout grid. They take the current text color unless `className` sets one.
export const Corners = ({ className, positions = CORNER_POSITIONS }: CornersProps): React.ReactElement[] =>
  positions.map((position) => (
    <svg aria-hidden key={position} viewBox="0 0 7 7" className={cn("absolute h-[7px] w-[7px]", position, className)}>
      <path d="M3.5 0v7M0 3.5h7" stroke="currentColor" strokeWidth="1" />
    </svg>
  ));
