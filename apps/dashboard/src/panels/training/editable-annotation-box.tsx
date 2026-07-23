import {
  annotationBBox,
  bboxToPercent,
  chromeScale,
  HANDLES,
  type BBox,
} from "@/panels/training/box-geometry";
import { BoxLabel } from "@/panels/training/box-label";
import { classColor } from "@/panels/training/class-color";
import { useBoxEdit } from "@/panels/training/use-box-edit";
import type { AnnotationDto, ImageRecordDto } from "@/lib/training-api";

/** On-screen size of a resize handle, before zoom counter-scaling. */
const HANDLE_PX = 11;

/**
 * The selected pending box with drag-to-move + 8 resize handles. The live `draft`
 * from `useBoxEdit` drives the outline; geometry PATCHes once on release. Only this
 * box takes pointer events, so pan/zoom stay live everywhere else.
 */
export function EditableAnnotationBox(props: {
  readonly annotation: AnnotationDto;
  readonly record: ImageRecordDto;
  readonly content: React.RefObject<HTMLDivElement | null>;
  readonly zoom: number;
  readonly onSetGeometry: (id: number, box: BBox) => void;
}): React.ReactElement | null {
  const { annotation, record, content, zoom, onSetGeometry } = props;
  const committed = annotationBBox(annotation);

  const { draft, startDrag } = useBoxEdit({
    content,
    record,
    bbox: committed,
    onCommit: (next) => {
      if (annotation.id !== null) {
        onSetGeometry(annotation.id, next);
      }
    },
  });

  // Editing is bbox-only and a persisted box always has an id; bail after the hook
  // (rules-of-hooks) for the cases the lightbox already filters out.
  if (annotation.geom_type !== "bbox" || annotation.id === null) {
    return null;
  }

  const box = draft ?? committed;
  const color = classColor(annotation.class_id);
  const chrome = chromeScale(zoom);
  const handleSize = HANDLE_PX * chrome;

  return (
    <div className="absolute" style={bboxToPercent(box, record)}>
      {/* Body: drag anywhere inside to move the whole box. */}
      <div
        className="absolute inset-0"
        style={{
          borderWidth: `${String(2 * chrome)}px`,
          borderStyle: "solid",
          borderColor: color,
          backgroundColor: classColor(annotation.class_id, 0.12),
          cursor: "move",
          touchAction: "none",
        }}
        onPointerDown={(event) => {
          startDrag("move", event);
        }}
      />

      {HANDLES.map((handle) => (
        <div
          key={handle.id}
          aria-label={`Resize ${handle.id}`}
          className="absolute rounded-[2px] border border-white"
          style={{
            left: `${String(handle.fx * 100)}%`,
            top: `${String(handle.fy * 100)}%`,
            width: `${String(handleSize)}px`,
            height: `${String(handleSize)}px`,
            transform: "translate(-50%, -50%)",
            backgroundColor: color,
            cursor: handle.cursor,
            touchAction: "none",
          }}
          onPointerDown={(event) => {
            startDrag(handle.id, event);
          }}
        />
      ))}

      <BoxLabel color={color} chrome={chrome} name={annotation.class_name} />
    </div>
  );
}
