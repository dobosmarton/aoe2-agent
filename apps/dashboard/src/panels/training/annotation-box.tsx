import { classColor } from "@/panels/training/class-color";
import type { AnnotationDto, ImageRecordDto } from "@/lib/training-api";

type Box = {
  readonly left: string;
  readonly top: string;
  readonly width: string;
  readonly height: string;
};

/** Percentage box for one annotation, relative to the image's natural size.
 * Percentages (not pixels) mean the overlay tracks whatever size the browser
 * settles on for the image — no measuring, no resize listener. */
const boxPercent = (ann: AnnotationDto, record: ImageRecordDto): Box => {
  const pct = (value: number, extent: number): string =>
    `${String((value / extent) * 100)}%`;

  if (ann.geom_type === "bbox") {
    const [x, y, w, h] = ann.coords;
    return {
      left: pct(x, record.width),
      top: pct(y, record.height),
      width: pct(w, record.width),
      height: pct(h, record.height),
    };
  }

  const xs = ann.coords.map(([x]) => x);
  const ys = ann.coords.map(([, y]) => y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  return {
    left: pct(minX, record.width),
    top: pct(minY, record.height),
    width: pct(Math.max(...xs) - minX, record.width),
    height: pct(Math.max(...ys) - minY, record.height),
  };
};

/** One annotation drawn over the image, dimmed when the legend focuses another
 * class. */
export function AnnotationBox(props: {
  readonly annotation: AnnotationDto;
  readonly record: ImageRecordDto;
  readonly zoom: number;
  /** null = nothing hovered; otherwise the class the legend is pointing at. */
  readonly focus: string | null;
}): React.ReactElement {
  const { annotation, record, zoom, focus } = props;
  const on = focus === null || focus === annotation.class_name;
  const color = classColor(annotation.class_id);
  // Chrome would otherwise render label and border `zoom`× bigger. Dividing the
  // zoom straight out pins them to a constant on-screen size, which reads as
  // *shrinking* text against an ever-growing image — so let them grow, but
  // sublinearly and capped, so they never swallow the object they annotate.
  const chrome = Math.min(zoom ** 0.5, 2.2) / zoom;

  return (
    <div
      className="pointer-events-none absolute transition-opacity duration-150"
      style={{
        ...boxPercent(annotation, record),
        border: `${String((on && focus !== null ? 3 : 2) * chrome)}px solid ${color}`,
        // A tint on the focused class makes a lone box findable in a busy frame.
        backgroundColor:
          on && focus !== null ? classColor(annotation.class_id, 0.18) : undefined,
        opacity: on ? 1 : 0.12,
      }}
    >
      <span
        className="absolute bottom-full left-0 whitespace-nowrap px-1 font-mono text-[11px] leading-tight text-black"
        style={{
          backgroundColor: color,
          transform: `scale(${String(chrome)})`,
          transformOrigin: "left bottom",
        }}
      >
        {annotation.class_name}
      </span>
    </div>
  );
}
