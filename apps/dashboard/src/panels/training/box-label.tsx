/** The class-name chip above a box, counter-scaled so it stays a steady on-screen
 * size as the image zooms. Shared by the read-only and editable overlays. */
export function BoxLabel(props: {
  readonly color: string;
  readonly chrome: number;
  readonly name: string;
}): React.ReactElement {
  const { color, chrome, name } = props;
  return (
    <span
      className="pointer-events-none absolute bottom-full left-0 whitespace-nowrap px-1 font-mono text-[11px] leading-tight text-black"
      style={{
        backgroundColor: color,
        transform: `scale(${String(chrome)})`,
        transformOrigin: "left bottom",
      }}
    >
      {name}
    </span>
  );
}
