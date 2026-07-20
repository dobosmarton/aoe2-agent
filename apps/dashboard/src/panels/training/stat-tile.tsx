import { Card, CardContent } from "@/components/ui/card";

/** One headline number in the coverage summary row. */
export function StatTile(props: {
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
