import { annotationBBox, bboxToPercent, chromeScale } from "@/panels/training/box-geometry";
import { BoxLabel } from "@/panels/training/box-label";
import { classColor } from "@/panels/training/class-color";
import type { AnnotationDto, ImageRecordDto } from "@/lib/training-api";

/** One annotation drawn over the image, dimmed when the legend focuses another class. */
export function AnnotationBox(props: {
  readonly annotation: AnnotationDto;
  readonly record: ImageRecordDto;
  readonly zoom: number;
  /** Class name the legend is focusing, or null when nothing is hovered. */
  readonly focus: string | null;
}): React.ReactElement {
  const { annotation, record, zoom, focus } = props;
  const on = focus === null || focus === annotation.class_name;
  const color = classColor(annotation.class_id);
  const chrome = chromeScale(zoom);

  return (
    <div
      className="pointer-events-none absolute transition-opacity duration-150"
      style={{
        ...bboxToPercent(annotationBBox(annotation), record),
        borderWidth: `${String((on && focus !== null ? 3 : 2) * chrome)}px`,
        // Dashed = a model prediction awaiting review; solid = approved/hand-drawn.
        borderStyle: annotation.status === "pending" ? "dashed" : "solid",
        borderColor: color,
        // Tint the focused class so a lone box is findable in a busy frame.
        backgroundColor:
          on && focus !== null ? classColor(annotation.class_id, 0.18) : undefined,
        opacity: on ? 1 : 0.12,
      }}
    >
      <BoxLabel color={color} chrome={chrome} name={annotation.class_name} />
    </div>
  );
}
