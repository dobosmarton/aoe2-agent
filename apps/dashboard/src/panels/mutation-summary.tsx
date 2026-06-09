import { Badge } from "@/components/ui/badge";
import { MUTATION_FIELDS, RESOURCE_COLORS } from "@/panels/operator-fields";
import type { MutationPatch } from "@/lib/api";

/** Chips summarising which fields a fork's mutation patch overrides. */
export function MutationSummary({
  mutation,
}: {
  readonly mutation: MutationPatch;
}): React.ReactElement {
  const present = MUTATION_FIELDS.filter((field) => mutation[field] !== undefined);
  if (present.length === 0) {
    return (
      <span className="text-muted-foreground text-xs">
        No overrides — clean clone of the parent state.
      </span>
    );
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {present.map((field) => {
        const colorVar = RESOURCE_COLORS[field];
        const value = mutation[field];
        return (
          <Badge key={field} variant="secondary" className="gap-1.5 font-mono">
            {colorVar === undefined ? null : (
              <span
                className="inline-block size-2 rounded-[2px]"
                style={{ backgroundColor: colorVar }}
              />
            )}
            {field === "age"
              ? `age → ${String(value)}`
              : `${field} ${String(value)}`}
          </Badge>
        );
      })}
    </div>
  );
}
