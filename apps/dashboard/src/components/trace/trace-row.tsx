import { useState } from "react";
import { ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { JsonView } from "@/components/trace/json-view";
import { eventKindMeta, eventMetaTag, summarise } from "@/lib/event-meta";
import type { ArenaEvent } from "@/lib/events";
import { cn } from "@/lib/utils";

type TraceRowProps = {
  readonly event: ArenaEvent;
}

// A dense, color-coded trace event that expands to show its full JSON payload —
// the Grafana/Langfuse pattern. The collapsed line carries a kind icon + label,
// a one-line summary, and an optional right-aligned meta tag (e.g. cost).
export function TraceRow({ event }: TraceRowProps): React.ReactElement {
  const [open, setOpen] = useState(false);
  const meta = eventKindMeta[event.kind];
  const Icon = meta.icon;
  const tag = eventMetaTag(event);

  return (
    <Collapsible
      isExpanded={open}
      onExpandedChange={setOpen}
      className="border-border/60 border-l-2"
      style={{ borderLeftColor: meta.colorVar }}
    >
      <CollapsibleTrigger className="hover:bg-accent/40 group flex w-full items-center gap-2 px-2 py-1 text-left text-xs">
        <ChevronRight
          className={cn(
            "text-muted-foreground size-3 shrink-0 transition-transform",
            open && "rotate-90",
          )}
        />
        <Icon className="size-3.5 shrink-0" style={{ color: meta.colorVar }} />
        <span
          className="shrink-0 font-mono font-medium"
          style={{ color: meta.colorVar }}
        >
          {meta.label}
        </span>
        <span className="text-foreground/70 truncate font-mono">
          {summarise(event)}
        </span>
        {tag === null ? null : (
          <span className="text-muted-foreground ml-auto shrink-0 font-mono tabular-nums">
            {tag}
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="bg-muted/30 mx-2 mb-1.5 overflow-x-auto rounded-md p-2">
          <JsonView value={event} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
