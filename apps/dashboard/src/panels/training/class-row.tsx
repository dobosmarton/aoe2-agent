import { AlertTriangle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { ClassCoverageDto } from "@/lib/training-api";

/** One class's real/synthetic instance counts, as a bar relative to the most
 * covered class. */
export function ClassRow(props: {
  readonly klass: ClassCoverageDto;
  /** Instance count of the best-covered class, which sets the bar's full width. */
  readonly maxReal: number;
}): React.ReactElement {
  const { klass, maxReal } = props;
  const pct = maxReal === 0 ? 0 : Math.round((klass.real_instances / maxReal) * 100);

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 py-1.5">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate font-mono text-xs">{klass.name}</span>
          {klass.real_instances === 0 ? (
            <Badge variant="destructive" className="gap-1 text-[10px]">
              <AlertTriangle className="size-3" />0 real
            </Badge>
          ) : null}
        </div>
        {/* react-aria's ProgressBar has no visible label here, so it needs an
            explicit accessible name — it warns at runtime otherwise. */}
        <Progress value={pct} aria-label={`${klass.name} coverage`} className="mt-1 h-1.5" />
      </div>
      <div className="text-muted-foreground text-right font-mono text-xs tabular-nums">
        <span className="text-foreground">{klass.real_instances}</span>
        <span className="mx-1">real</span>
        <span>/ {klass.synth_instances} synth</span>
      </div>
    </div>
  );
}
