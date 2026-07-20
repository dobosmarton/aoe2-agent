/**
 * The two-column application frame: a fixed-width sidebar beside a scrollable
 * main column, pinned to the viewport.
 *
 * The `grid-rows-[1fr]` is the whole reason this component exists. An implicit
 * grid row is sized `auto`, which grows to fit its tallest child — `h-screen`
 * caps the *container*, not the row, so a tall panel pushes the row past the
 * viewport and no descendant can ever scroll. Both layouts hand-rolled this
 * frame and both got it wrong; the constraint now has one owner.
 */
export function AppShell(props: {
  /** Sidebar width in pixels. */
  readonly sidebarWidth: number;
  readonly sidebar: React.ReactNode;
  readonly children: React.ReactNode;
}): React.ReactElement {
  return (
    <div
      className="grid h-screen grid-rows-[1fr]"
      // Not a Tailwind arbitrary value: those are compiled from source text, so
      // `grid-cols-[${w}px_1fr]` would never be generated.
      style={{ gridTemplateColumns: `${String(props.sidebarWidth)}px 1fr` }}
    >
      <aside className="border-border bg-card flex min-h-0 flex-col border-r">
        {props.sidebar}
      </aside>
      <main className="bg-background flex min-h-0 min-w-0 flex-col">{props.children}</main>
    </div>
  );
}
