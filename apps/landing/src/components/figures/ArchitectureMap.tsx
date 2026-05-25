import * as React from "react";

/**
 * Hand-authored React + SVG hero diagram for the landing page.
 *
 * Mirrors the conceptual layout from docs/index.md's mermaid graph but as a
 * first-class interactive figure: each node is a link to the relevant Part,
 * and hovering a node highlights its outbound arrows so newcomers can see
 * the dataflow at a glance.
 *
 * Why an island, not an Astro <svg>: we need per-node hover state. The
 * actual SVG payload is tiny (no images, just shapes + text), so the JS
 * cost is negligible.
 */
export default function ArchitectureMap(): React.ReactElement {
  const [active, setActive] = React.useState<string | null>(null);

  // Each node carries its outgoing edge targets so the hover highlight is
  // declarative rather than encoded in CSS selectors per pair.
  const nodes: Record<
    string,
    { x: number; y: number; w: number; h: number; label: string; sub: string; href: string; out: string[] }
  > = {
    screen: { x: 30, y: 220, w: 130, h: 56, label: "Screenshot", sub: "mss capture", href: "/docs/part1-architecture/02-game-loop-pipeline", out: ["yolo", "strategist"] },
    yolo: { x: 200, y: 130, w: 130, h: 56, label: "YOLO v5", sub: "60-class detect", href: "/docs/part3-entity-detection/07-detector-architecture", out: ["executor"] },
    strategist: { x: 200, y: 310, w: 130, h: 56, label: "Strategist", sub: "Sonnet · vision", href: "/docs/part2-llm-integration/04-provider-pattern", out: ["executor"] },
    executor: { x: 370, y: 220, w: 130, h: 56, label: "Executor", sub: "Haiku · text", href: "/docs/part2-llm-integration/05-prompt-engineering", out: ["actions"] },
    actions: { x: 540, y: 220, w: 130, h: 56, label: "Actions", sub: "mouse / keyboard", href: "/docs/part1-architecture/03-action-model-and-execution", out: ["broker"] },
    broker: { x: 540, y: 410, w: 130, h: 56, label: "Event broker", sub: "in-proc / Redis", href: "/docs/part6-evaluation-arena/15-event-broker", out: ["duckdb", "arenaweb"] },
    duckdb: { x: 370, y: 410, w: 130, h: 56, label: "DuckDB log", sub: "replay + fork", href: "/docs/part6-evaluation-arena/16-duckdb-persister-and-replay", out: ["arena"] },
    arena: { x: 200, y: 410, w: 130, h: 56, label: "Arena CLI", sub: "race / rank", href: "/docs/part6-evaluation-arena/14-arena-overview", out: ["broker"] },
    arenaweb: { x: 710, y: 410, w: 130, h: 56, label: "Arena Web", sub: "FastAPI + SSE", href: "/docs/part7-arena-web/19-web-architecture", out: [] },
    autoresearch: { x: 710, y: 220, w: 130, h: 56, label: "Autoresearch", sub: "mutate → run → score", href: "/docs/part8-autoresearch/22-autoresearch-overview", out: ["strategist"] },
  };

  const isHighlighted = (from: string, to: string): boolean => {
    if (active === null) return false;
    if (active === from) return true;
    if (active === to) return true;
    return false;
  };

  return (
    <div className="w-full overflow-x-auto rounded-xl border bg-card p-4">
      <svg
        viewBox="0 0 880 500"
        role="img"
        aria-label="System architecture: capture, detect, think, act loop plus arena and autoresearch tiers"
        className="block w-full"
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
          <marker id="arrow-hi" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-foreground)" />
          </marker>
        </defs>

        {/* Tier labels */}
        <text x="20" y="100" className="fill-muted-foreground" fontSize="11" fontFamily="ui-sans-serif">REAL-GAME TIER</text>
        <text x="20" y="395" className="fill-muted-foreground" fontSize="11" fontFamily="ui-sans-serif">ARENA TIER</text>
        <line x1="20" y1="110" x2="860" y2="110" stroke="var(--color-border)" strokeDasharray="2 4" />
        <line x1="20" y1="380" x2="860" y2="380" stroke="var(--color-border)" strokeDasharray="2 4" />

        {/* Edges */}
        {Object.entries(nodes).flatMap(([fromId, from]) =>
          from.out.map((toId) => {
            const to = nodes[toId];
            if (!to) return null;
            const x1 = from.x + from.w;
            const y1 = from.y + from.h / 2;
            const x2 = to.x;
            const y2 = to.y + to.h / 2;
            const hi = isHighlighted(fromId, toId);
            return (
              <line
                key={`${fromId}-${toId}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={hi ? "var(--color-foreground)" : "var(--color-border)"}
                strokeWidth={hi ? 2 : 1.2}
                markerEnd={hi ? "url(#arrow-hi)" : "url(#arrow)"}
                className="transition-all"
                style={{ color: "var(--color-muted-foreground)" }}
              />
            );
          }),
        )}

        {/* Nodes */}
        {Object.entries(nodes).map(([id, n]) => {
          const isActive = active === id;
          return (
            <a key={id} href={n.href} onMouseEnter={() => setActive(id)} onMouseLeave={() => setActive(null)} onFocus={() => setActive(id)} onBlur={() => setActive(null)}>
              <rect
                x={n.x}
                y={n.y}
                width={n.w}
                height={n.h}
                rx={8}
                fill="var(--color-card)"
                stroke={isActive ? "var(--color-foreground)" : "var(--color-border)"}
                strokeWidth={isActive ? 2 : 1}
                className="transition-all cursor-pointer"
              />
              <text x={n.x + n.w / 2} y={n.y + 24} textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--color-foreground)" fontFamily="ui-sans-serif" className="pointer-events-none">
                {n.label}
              </text>
              <text x={n.x + n.w / 2} y={n.y + 42} textAnchor="middle" fontSize="10" fill="var(--color-muted-foreground)" fontFamily="ui-sans-serif" className="pointer-events-none">
                {n.sub}
              </text>
            </a>
          );
        })}
      </svg>
      <p className="mt-2 text-xs text-muted-foreground">
        Hover a node to see its outgoing dataflow. Click to jump to the relevant Part.
      </p>
    </div>
  );
}
