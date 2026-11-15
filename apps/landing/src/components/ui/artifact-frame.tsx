import * as React from "react";

import { cn } from "@/lib/utils";
import { Corners } from "@/components/ui/frame";

type ArtifactFrameProps = {
  // What the artifact holds (a table's subject, a chart title, an image's alt text).
  label: string;
  // A size-like detail shown in the accent color, like a detector's confidence score.
  meta?: string;
  // The explanation. When present it replaces the label in the bracketed text, since
  // the two usually say the same thing.
  caption?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

// The detector frame around artifacts (tables, code, charts, images): a thin outline
// with "+" corner marks holding only the artifact. Underneath, one mono line of text
// (the caption, or the label when there is none) and the meta, set in drawn [ ]
// brackets that stretch to fit several lines. Styles live in styles/globals.css under
// `.artifact`; lib/rehype-artifacts.ts writes the same markup for the docs.
export function ArtifactFrame({ label, meta, caption, children, className }: ArtifactFrameProps) {
  return (
    <figure className={cn("artifact", className)}>
      <div className="artifact-frame">
        <Corners />
        {children}
      </div>
      <figcaption className="artifact-caption">
        {caption ?? label}
        {meta && <span className="artifact-label-meta"> {meta}</span>}
      </figcaption>
    </figure>
  );
}
