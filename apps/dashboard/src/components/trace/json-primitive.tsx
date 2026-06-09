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
    default:
      return <span className="text-foreground">{String(value)}</span>;
  }
}
