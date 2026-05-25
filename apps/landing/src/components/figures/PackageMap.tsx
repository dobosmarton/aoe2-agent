import * as React from "react";
import { ArrowRight, Box } from "lucide-react";

import { PACKAGES } from "@/lib/packages";
import { PARTS } from "@/lib/taxonomy";
import { cn } from "@/lib/utils";

const TIER_LABELS: Record<string, string> = {
  shared: "Shared",
  "real-game": "Real game",
  detection: "Detection",
  arena: "Arena",
  ops: "Operations",
};

const TIER_ORDER = ["shared", "detection", "real-game", "arena", "ops"];

/**
 * Card grid for the 9-package uv workspace. Hovering a card dims unrelated
 * packages and highlights its direct dependencies.
 *
 * React island because of the hover-link visualization. Without that, this
 * could be a pure Astro component.
 */
export default function PackageMap(): React.ReactElement {
  const [hovered, setHovered] = React.useState<string | null>(null);

  const tiers = TIER_ORDER.map((tier) => ({
    tier,
    packages: PACKAGES.filter((p) => p.tier === tier),
  })).filter((g) => g.packages.length > 0);

  const dependsByHovered = hovered
    ? new Set(PACKAGES.find((p) => p.id === hovered)?.dependsOn ?? [])
    : null;

  return (
    <div className="space-y-6">
      {tiers.map(({ tier, packages }) => (
        <section key={tier}>
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {TIER_LABELS[tier]}
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {packages.map((pkg) => {
              const part = pkg.part
                ? PARTS.find((p) => p.id === pkg.part)
                : undefined;
              const isHovered = hovered === pkg.id;
              const isDependency =
                dependsByHovered?.has(pkg.id) ?? false;
              const dimmed =
                hovered !== null && !isHovered && !isDependency;
              return (
                <article
                  key={pkg.id}
                  onMouseEnter={() => setHovered(pkg.id)}
                  onMouseLeave={() => setHovered(null)}
                  className={cn(
                    "group relative rounded-xl border bg-card p-4 transition-all",
                    isHovered && "border-foreground shadow-md",
                    isDependency && "border-foreground/60 bg-accent/40",
                    dimmed && "opacity-40",
                  )}
                >
                  <div className="mb-2 flex items-center gap-2">
                    <Box className="h-4 w-4 text-muted-foreground" />
                    <h3 className="font-mono text-sm font-semibold">
                      {pkg.title}
                    </h3>
                  </div>
                  <p className="mb-3 text-sm text-muted-foreground">
                    {pkg.blurb}
                  </p>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {pkg.dependsOn.length > 0 && (
                      <span className="text-muted-foreground">
                        depends on{" "}
                        {pkg.dependsOn.map((d, i) => (
                          <span key={d}>
                            <code className="rounded bg-muted px-1 py-0.5 font-mono">
                              {d}
                            </code>
                            {i < pkg.dependsOn.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </span>
                    )}
                    {part && (
                      <a
                        href={`/docs/${part.id}/${part.chapters[0]!.slug}`}
                        className="ml-auto inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
                      >
                        Part {part.label}
                        <ArrowRight className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}
