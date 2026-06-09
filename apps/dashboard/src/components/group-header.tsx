import { Activity, ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { LiveDot } from "@/components/live-dot";
import { cn } from "@/lib/utils";
import { formatRelative, labelVariant } from "@/lib/run-format";
import type { RunGroup } from "@/lib/run-grouping";

interface GroupHeaderProps {
  readonly group: RunGroup;
  readonly expanded: boolean;
  readonly active: boolean;
  readonly onToggle: (key: string) => void;
  readonly onOpen: (key: string) => void;
}

/** Header for a multi-run operation. The chevron toggles the nested runs; the
 * title opens the experiment overview for the operation. Two separate buttons
 * (no nesting) so each affordance is independently clickable. */
export function GroupHeader({
  group,
  expanded,
  active,
  onToggle,
  onOpen,
}: GroupHeaderProps): React.ReactElement {
  return (
    <div
      className={cn(
        "flex w-full items-center gap-1 rounded-md pr-2 transition-colors",
        active ? "bg-accent/60" : "hover:bg-accent/50",
      )}
    >
      <button
        type="button"
        aria-label={expanded ? "Collapse runs" : "Expand runs"}
        onClick={() => {
          onToggle(group.key);
        }}
        className="flex shrink-0 items-center py-1.5 pl-2"
      >
        <ChevronRight
          className={cn(
            "text-muted-foreground size-3.5 transition-transform",
            expanded && "rotate-90",
          )}
        />
      </button>
      <button
        type="button"
        title="Open experiment overview"
        onClick={() => {
          onOpen(group.key);
        }}
        className="flex flex-1 items-center gap-1.5 py-1.5 text-left"
      >
        <Badge variant={labelVariant(group.label)}>{group.label}</Badge>
        {group.running ? <LiveDot /> : null}
        <span className="text-muted-foreground ml-auto flex items-center gap-2 text-xs tabular-nums">
          <span title={group.firstTs}>{formatRelative(group.lastTs)}</span>
          <span>{group.runs.length} runs</span>
          <span className="flex items-center gap-1">
            <Activity className="size-3" />
            {group.totalEvents}
          </span>
        </span>
      </button>
    </div>
  );
}
