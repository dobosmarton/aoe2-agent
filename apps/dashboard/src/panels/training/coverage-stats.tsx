import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Database, ImageIcon, Tags } from "lucide-react";

import { QueryFallback } from "@/components/query-fallback";
import { Card, CardContent } from "@/components/ui/card";
import { trackerDatasetsQueryOptions, trackerStatsQueryOptions } from "@/lib/queries";
import { ClassRow } from "@/panels/training/class-row";
import { StatTile } from "@/panels/training/stat-tile";

export function CoverageStats(): React.ReactElement {
  const statsQuery = useQuery(trackerStatsQueryOptions());
  const datasetsQuery = useQuery(trackerDatasetsQueryOptions());
  const stats = statsQuery.data;

  if (stats === undefined) {
    return <QueryFallback noun="coverage" query={statsQuery} className="p-6" />;
  }

  const sorted = [...stats.classes].sort((a, b) => b.real_instances - a.real_instances);
  const maxReal = sorted[0]?.real_instances ?? 0;
  const latestDataset = datasetsQuery.data?.[0] ?? null;

  return (
    // Block, not flex-col: a flex child shrinks by default, and Card is
    // `overflow-hidden`, so in a bounded flex column the class list gets
    // squashed and clipped instead of scrolling the pane.
    <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-6">
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
              {latestDataset.n_real_images} real · {latestDataset.n_synth_images} synthetic
              images
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
