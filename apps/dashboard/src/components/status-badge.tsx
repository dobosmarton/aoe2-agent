import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { SseStatus } from "@/hooks/use-events";

function statusLabel(status: SseStatus): string {
  switch (status) {
    case "idle":
      return "Idle";
    case "connecting":
      return "Connecting…";
    case "open":
      return "Streaming";
    case "closed":
      return "Complete";
    case "error":
      return "Error";
  }
}

/** SSE connection status for the selected run, with a pulse while streaming. */
export function StatusBadge({ status }: { readonly status: SseStatus }): React.ReactElement {
  const streaming = status === "open";
  return (
    <Badge
      variant={status === "error" ? "destructive" : "outline"}
      className={cn(
        "gap-1.5",
        streaming && "border-emerald-500/40 text-emerald-600 dark:text-emerald-400",
      )}
    >
      {streaming ? (
        <span className="relative flex size-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-75" />
          <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
        </span>
      ) : null}
      {statusLabel(status)}
    </Badge>
  );
}
