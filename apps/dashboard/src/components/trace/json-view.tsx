import { JsonNode } from "@/components/trace/json-node";

// ---------------------------------------------------------------------------
// Lightweight, themed JSON renderer. Primitives are colored by type; objects
// and arrays are collapsible. Kept in-repo (vs. a react-json-view dependency)
// so it matches the dark theme and stays strict-TS friendly. Event payloads
// are shallow, so no virtualization is needed.
// ---------------------------------------------------------------------------

interface JsonViewProps {
  readonly value: unknown;
  readonly defaultExpanded?: boolean;
}

export function JsonView({
  value,
  defaultExpanded = true,
}: JsonViewProps): React.ReactElement {
  return (
    <div className="font-mono text-[11px] leading-relaxed">
      <JsonNode value={value} depth={0} expandDepth={defaultExpanded ? 2 : 1} />
    </div>
  );
}
