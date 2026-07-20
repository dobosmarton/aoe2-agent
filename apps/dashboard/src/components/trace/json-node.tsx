import { useState } from "react";
import { ChevronRight } from "lucide-react";

import { JsonPrimitive } from "@/components/trace/json-primitive";
import { cn } from "@/lib/utils";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type JsonNodeProps = {
  readonly name?: string;
  readonly value: unknown;
  readonly depth: number;
  readonly expandDepth: number;
}

/** One node of the JSON tree — a primitive line, or a collapsible object/array
 * that recurses into child `JsonNode`s. */
export function JsonNode({
  name,
  value,
  depth,
  expandDepth,
}: JsonNodeProps): React.ReactElement {
  const complex = isPlainObject(value) || Array.isArray(value);
  const [open, setOpen] = useState(depth < expandDepth);
  const indent = { paddingLeft: `${String(depth * 12)}px` };

  if (!complex) {
    return (
      <div style={indent} className="whitespace-pre">
        {name === undefined ? null : (
          <span className="text-muted-foreground">{name}: </span>
        )}
        <JsonPrimitive value={value} />
      </div>
    );
  }

  const entries: ReadonlyArray<readonly [string, unknown]> = Array.isArray(value)
    ? value.map((v, i) => [String(i), v] as const)
    : Object.entries(value);
  const openBrace = Array.isArray(value) ? "[" : "{";
  const closeBrace = Array.isArray(value) ? "]" : "}";

  return (
    <div>
      <div
        style={indent}
        className="hover:bg-accent/30 flex cursor-pointer items-center gap-0.5 whitespace-pre rounded-sm"
        onClick={() => {
          setOpen((v) => !v);
        }}
      >
        <ChevronRight
          className={cn(
            "text-muted-foreground size-3 shrink-0 transition-transform",
            open && "rotate-90",
          )}
        />
        {name === undefined ? null : (
          <span className="text-muted-foreground">{name}: </span>
        )}
        <span className="text-muted-foreground">
          {open ? openBrace : `${openBrace} … ${closeBrace} ${String(entries.length)}`}
        </span>
      </div>
      {open ? (
        <>
          {entries.map(([k, v]) => (
            <JsonNode
              key={k}
              name={k}
              value={v}
              depth={depth + 1}
              expandDepth={expandDepth}
            />
          ))}
          <div style={indent} className="text-muted-foreground whitespace-pre">
            {closeBrace}
          </div>
        </>
      ) : null}
    </div>
  );
}
