import { cn } from "@/lib/utils";
import { errorMessage } from "@/lib/load-status";

/** The part of a TanStack query this needs — a plain shape so callers can pass
 * a query object directly without this module depending on Query's generics. */
type QueryLike = {
  readonly isPending: boolean;
  readonly isError: boolean;
  readonly error: unknown;
};

/**
 * What to render *instead of* a panel's content while its query is pending or
 * failed, or `null` once there is data to show.
 *
 * `noun` keeps the copy specific ("Failed to load coverage: …") while the shape
 * of the message lives in one place; the three training panels previously
 * carried three hand-written copies of this pair.
 */
export function QueryFallback(props: {
  readonly noun: string;
  readonly query: QueryLike;
  /** Padding differs by host — the panels use `p-6`, the lightbox none. */
  readonly className?: string;
}): React.ReactElement | null {
  const { noun, query, className } = props;

  if (query.isPending) {
    return (
      <p className={cn("text-muted-foreground text-sm", className)}>Loading {noun}…</p>
    );
  }
  if (query.isError) {
    return (
      <p className={cn("text-destructive text-sm", className)}>
        Failed to load {noun}: {errorMessage(query.error) ?? "unknown error"}
      </p>
    );
  }
  return null;
}
