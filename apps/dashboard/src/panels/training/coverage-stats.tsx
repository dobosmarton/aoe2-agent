import { AlertTriangle, Database, ImageIcon, Tags } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useQuery } from "@tanstack/react-query";
import {
  trackerDatasetsQueryOptions,
  trackerStatsQueryOptions,
} from "@/lib/queries";
import { errorMessage } from "@/lib/load-status";
import type { ClassCoverageDto } from "@/lib/training-api";

function StatTile(props: {
  readonly icon: React.ReactNode;
  readonly label: string;
  readonly value: string;
}): React.ReactElement {
  return (
    <Card className="gap-1 py-4">
      <CardContent className="flex items-center gap-3 px-4">
        <div className="text-muted-foreground">{props.icon}</div>
        <div className="min-w-0">
          <div className="text-2xl font-semibold tabular-nums">{props.value}</div>
          <div className="text-muted-foreground text-xs">{props.label}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function ClassRow(props: {
  readonly klass: ClassCoverageDto;
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

export function CoverageStats(): React.ReactElement {
  const statsQuery = useQuery(trackerStatsQueryOptions());
  const datasetsQuery = useQuery(trackerDatasetsQueryOptions());
  const stats = statsQuery.data;

  if (statsQuery.isPending) {
    return <p className="text-muted-foreground p-6 text-sm">Loading coverage…</p>;
  }
  if (statsQuery.isError || stats === undefined) {
    return (
      <p className="text-destructive p-6 text-sm">
        Failed to load coverage: {errorMessage(statsQuery.error) ?? "unknown error"}
      </p>
    );
  }

  const sorted = [...stats.classes].sort((a, b) => b.real_instances - a.real_instances);
  const maxReal = sorted[0]?.real_instances ?? 0;
  const latestDataset = datasetsQuery.data?.[0] ?? null;

  return (
    <div className="flex min-h-0 flex-col gap-4 overflow-auto p-6">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          icon={<ImageIcon className="size-5" />}
          label="Screenshots"
          value={String(stats.total_images)}
        />
        <StatTile
          icon={<Tags className="size-5" />}
          label="Labeled"
          value={String(stats.labeled_images)}
        />
        <StatTile
          icon={<ImageIcon className="size-5" />}
          label="Unlabeled"
          value={String(stats.unlabeled_images)}
        />
        <StatTile
          icon={<AlertTriangle className="size-5" />}
          label="Classes with no real labels"
          value={String(stats.zero_real_class_ids.length)}
        />
      </div>

      {latestDataset !== null ? (
        <Card className="py-3">
          <CardContent className="flex flex-wrap items-center gap-2 px-4 text-xs">
            <Database className="text-muted-foreground size-4" />
            <span className="font-semibold">{latestDataset.name}</span>
            <span className="text-muted-foreground">
              {latestDataset.n_real_images} real · {latestDataset.n_synth_images} synthetic images
            </span>
          </CardContent>
        </Card>
      ) : null}

      <Card className="py-4">
        <CardContent className="px-4">
          <h2 className="mb-2 text-sm font-semibold">Per-class coverage</h2>
          <div className="divide-border divide-y">
            {sorted.map((klass) => (
              <ClassRow key={klass.class_id} klass={klass} maxReal={maxReal} />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
