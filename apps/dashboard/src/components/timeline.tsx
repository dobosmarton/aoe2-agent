import { Slider } from "@/components/ui/slider";

interface TimelineProps {
  readonly maxTurn: number | null;
  readonly selectedTurn: number | null;
  readonly onSelect: (turn: number) => void;
}

export function Timeline({
  maxTurn,
  selectedTurn,
  onSelect,
}: TimelineProps): React.ReactElement {
  if (maxTurn === null || maxTurn < 1) {
    return (
      <div className="text-muted-foreground border-border border-t px-4 py-3 text-xs">
        Timeline activates once turn events arrive…
      </div>
    );
  }

  const current = selectedTurn ?? maxTurn;

  return (
    <div className="border-border flex items-center gap-4 border-t px-4 py-3">
      <span className="text-muted-foreground font-mono text-xs whitespace-nowrap">
        turn {current} / {maxTurn}
      </span>
      <Slider
        min={1}
        max={maxTurn}
        step={1}
        value={[current]}
        onValueChange={(values) => {
          const next = values[0];
          if (next !== undefined) {
            onSelect(next);
          }
        }}
        className="flex-1"
      />
    </div>
  );
}
