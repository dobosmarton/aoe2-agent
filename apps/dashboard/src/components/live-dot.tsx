/** The pulsing emerald "in progress" indicator, shared by run cards and group
 * headers. */
export function LiveDot(): React.ReactElement {
  return (
    <span className="relative flex size-1.5">
      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-500 opacity-75" />
      <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
    </span>
  );
}
