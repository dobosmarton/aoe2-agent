import * as React from "react";
import { ArrowRight } from "lucide-react";

import { PACKAGES } from "@/lib/packages";
import { getPage } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";
import { CellFooter, CellHeader } from "@/components/ui/frame";

const TIER_LABELS: Record<string, string> = {
  shared: "Shared",
  "real-game": "Real game",
  detection: "Detection",
  arena: "Arena",
  ops: "Operations",
  reference: "Reference",
};

const TIER_ORDER = ["shared", "detection", "real-game", "arena", "ops", "reference"];

/**
 * One even cell grid for the uv workspace, in the blog's cell style, ordered by
 * tier. Hovering a cell dims unrelated packages and tints its direct
 * dependencies in the accent color.
 *
 * React island because of the hover-link visualization. Without that, this
 * could be a pure Astro component.
 */
export default function PackageMap(): React.ReactElement {
  const [hovered, setHovered] = React.useState<string | null>(null);

  // One even grid, ordered by tier; the tier name sits in each cell's header.
  const packages = [...PACKAGES].sort((a, b) => TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier));

  const dependsByHovered = hovered ? new Set(PACKAGES.find((p) => p.id === hovered)?.dependsOn ?? []) : null;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {packages.map((pkg) => {
        const docPage = pkg.doc ? getPage(pkg.doc) : undefined;
        const isHovered = hovered === pkg.id;
        const isDependency = dependsByHovered?.has(pkg.id) ?? false;
        const dimmed = hovered !== null && !isHovered && !isDependency;
        return (
          <article
            key={pkg.id}
            onMouseEnter={() => setHovered(pkg.id)}
            onMouseLeave={() => setHovered(null)}
            className={cn(
              "flex flex-col overflow-hidden rounded-lg border border-foreground/10 bg-card transition-all duration-200",
              isHovered && "border-accent-foreground/50 shadow-[0_12px_32px_-16px_rgb(0_0_0/0.25)]",
              isDependency && "border-accent-foreground/40 bg-accent-foreground/[0.06]",
              dimmed && "opacity-40",
            )}
          >
            <CellHeader
              label={pkg.title}
              aside={
                docPage ? (
                  <a
                    href={`/docs/${docPage.fullSlug}`}
                    title={docPage.page.title}
                    className="inline-flex items-center gap-1 transition-colors hover:text-accent-foreground"
                  >
                    {docPage.section.title}
                    <ArrowRight className="h-3 w-3" />
                  </a>
                ) : (
                  TIER_LABELS[pkg.tier]
                )
              }
            />
            <p className="flex-1 px-4 py-3 text-sm text-muted-foreground">{pkg.blurb}</p>
            <CellFooter>
              {pkg.dependsOn.length > 0 ? (
                <>
                  depends on{" "}
                  {pkg.dependsOn.map((d, i) => (
                    <span key={d}>
                      <span className={cn(isHovered && "text-accent-foreground")}>{d}</span>
                      {i < pkg.dependsOn.length - 1 ? ", " : ""}
                    </span>
                  ))}
                </>
              ) : (
                "no workspace dependencies"
              )}
            </CellFooter>
          </article>
        );
      })}
    </div>
  );
}
