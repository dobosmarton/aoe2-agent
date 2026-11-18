import * as React from "react";
import { ArrowRight } from "lucide-react";

import { PACKAGES, type PackageInfo } from "@/lib/packages";
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

const PACKAGES_BY_TIER = [...PACKAGES].sort((a, b) => TIER_ORDER.indexOf(a.tier) - TIER_ORDER.indexOf(b.tier));

// The package's guide section as a link, or its tier name when it has no guide page.
const packageAside = (pkg: PackageInfo): React.ReactNode => {
  const docPage = pkg.doc ? getPage(pkg.doc) : undefined;

  if (!docPage) {
    return TIER_LABELS[pkg.tier];
  }

  return (
    <a
      href={`/docs/${docPage.fullSlug}`}
      title={docPage.page.title}
      className="inline-flex items-center gap-1 transition-colors hover:text-accent-foreground"
    >
      {docPage.section.title}
      <ArrowRight className="h-3 w-3" />
    </a>
  );
};

const DependencyList = ({ dependsOn, highlighted }: { dependsOn: string[]; highlighted: boolean }): React.ReactNode => {
  if (dependsOn.length === 0) {
    return "no workspace dependencies";
  }

  return (
    <>
      depends on{" "}
      {dependsOn.map((dependency, index) => (
        <span key={dependency}>
          <span className={cn(highlighted && "text-accent-foreground")}>{dependency}</span>
          {index < dependsOn.length - 1 ? ", " : ""}
        </span>
      ))}
    </>
  );
};

type PackageCellProps = {
  pkg: PackageInfo;
  isHovered: boolean;
  isDependency: boolean;
  dimmed: boolean;
  onHover: (id: string | null) => void;
};

const PackageCell = ({ pkg, isHovered, isDependency, dimmed, onHover }: PackageCellProps): React.ReactElement => (
  <article
    onMouseEnter={() => onHover(pkg.id)}
    onMouseLeave={() => onHover(null)}
    className={cn(
      "flex flex-col overflow-hidden rounded-lg border border-foreground/10 bg-card transition-all duration-200",
      isHovered && "border-accent-foreground/50 shadow-[0_12px_32px_-16px_rgb(0_0_0/0.25)]",
      isDependency && "border-accent-foreground/40 bg-accent-foreground/[0.06]",
      dimmed && "opacity-40",
    )}
  >
    <CellHeader label={pkg.title} aside={packageAside(pkg)} />
    <p className="flex-1 px-4 py-3 text-sm text-muted-foreground">{pkg.blurb}</p>
    <CellFooter>
      <DependencyList dependsOn={pkg.dependsOn} highlighted={isHovered} />
    </CellFooter>
  </article>
);

/**
 * One even cell grid for the uv workspace, in the blog's cell style, ordered by
 * tier. Hovering a cell dims unrelated packages and tints its direct dependencies
 * in the accent color.
 *
 * React island because of the hover-link visualization. Without that, this
 * could be a pure Astro component.
 */
export const PackageMap = (): React.ReactElement => {
  const [hovered, setHovered] = React.useState<string | null>(null);
  const dependsByHovered = hovered ? new Set(PACKAGES.find((pkg) => pkg.id === hovered)?.dependsOn ?? []) : null;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {PACKAGES_BY_TIER.map((pkg) => {
        const isHovered = hovered === pkg.id;
        const isDependency = dependsByHovered?.has(pkg.id) ?? false;

        return (
          <PackageCell
            key={pkg.id}
            pkg={pkg}
            isHovered={isHovered}
            isDependency={isDependency}
            dimmed={hovered !== null && !isHovered && !isDependency}
            onHover={setHovered}
          />
        );
      })}
    </div>
  );
};
