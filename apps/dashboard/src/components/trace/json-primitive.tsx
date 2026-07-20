/** Renders a single JSON primitive value, colored by its type. */
export function JsonPrimitive({
  value,
}: {
  readonly value: unknown;
}): React.ReactElement {
  if (value === null) {
    return <span className="text-muted-foreground">null</span>;
  }
  switch (typeof value) {
    case "string":
      return <span className="text-food">"{value}"</span>;
    case "number":
      return <span className="text-gold tabular-nums">{String(value)}</span>;
    case "boolean":
      return <span className="text-event-llm-prompt">{String(value)}</span>;
    case "bigint":
      return <span className="text-gold tabular-nums">{value.toString()}</span>;
    case "undefined":
      return <span className="text-muted-foreground">undefined</span>;
    // Symbols and functions have no JSON form, and an object reaching a
    // *primitive* renderer means JsonNode mis-dispatched. Naming the kind beats
    // rendering "[object Object]" and pretending it was data.
    default:
      return <span className="text-muted-foreground">[{typeof value}]</span>;
  }
}
