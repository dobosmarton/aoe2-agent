import { useEffect } from "react";
import { X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useImageDetail } from "@/hooks/use-tracker";
import { trackerAssetUrl, type AnnotationDto, type ImageRecordDto } from "@/lib/training-api";

/** Golden-angle hue rotation: consecutive class ids land far apart on the wheel,
 * so neighbouring classes stay distinguishable without a hand-picked palette. */
function classColor(classId: number): string {
  return `hsl(${String((classId * 137.508) % 360)} 90% 60%)`;
}

/** Percentage box for one annotation, relative to the image's natural size.
 * Percentages (not pixels) mean the overlay tracks whatever size the browser
 * settles on for the image — no measuring, no resize listener. */
function boxPercent(
  ann: AnnotationDto,
  record: ImageRecordDto,
): { left: string; top: string; width: string; height: string } {
  const pct = (value: number, extent: number): string => `${String((value / extent) * 100)}%`;
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
}

function AnnotationBox(props: {
  readonly annotation: AnnotationDto;
  readonly record: ImageRecordDto;
}): React.ReactElement {
  const { annotation, record } = props;
  const color = classColor(annotation.class_id);
  return (
    <div
      className="pointer-events-none absolute border-2"
      style={{ ...boxPercent(annotation, record), borderColor: color }}
    >
      <span
        className="absolute left-0 top-0 -translate-y-full whitespace-nowrap px-1 font-mono text-[10px] leading-tight text-black"
        style={{ backgroundColor: color }}
      >
        {annotation.class_name}
      </span>
    </div>
  );
}

export function ImageLightbox(props: {
  readonly imageId: number | null;
  readonly onClose: () => void;
}): React.ReactElement | null {
  const { imageId, onClose } = props;
  const { data, status, error } = useImageDetail(imageId);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  if (imageId === null) {
    return null;
  }

  const classNames = data === null ? [] : [...new Set(data.annotations.map((a) => a.class_name))];

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Image detail"
      className="fixed inset-0 z-50 flex flex-col bg-black/80 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="border-border bg-card mx-auto flex max-h-full min-h-0 w-full max-w-[1400px] flex-col rounded-lg border shadow-2xl"
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        <header className="border-border flex items-center gap-3 border-b px-4 py-2">
          <span className="truncate font-mono text-xs" title={data?.image.filename}>
            {data?.image.filename ?? "Loading…"}
          </span>
          {data !== null ? (
            <span className="text-muted-foreground shrink-0 font-mono text-[11px] tabular-nums">
              {data.image.width}×{data.image.height} · {data.annotations.length} boxes
            </span>
          ) : null}
          <Button variant="ghost" size="sm" className="ml-auto shrink-0" onClick={onClose}>
            <X className="size-4" />
          </Button>
        </header>

        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-4">
          {status === "loading" ? (
            <p className="text-muted-foreground text-sm">Loading image…</p>
          ) : status === "error" || data === null ? (
            <p className="text-destructive text-sm">
              Failed to load image: {error ?? "unknown error"}
            </p>
          ) : (
            // Shrink-wraps the image so the percentage-positioned boxes align.
            <div className="relative inline-block">
              <img
                src={trackerAssetUrl(data.image.raw_url)}
                alt={data.image.filename}
                className="block max-h-[70vh] max-w-full object-contain"
              />
              {data.annotations.map((ann, index) => (
                <AnnotationBox
                  key={ann.id ?? index}
                  annotation={ann}
                  record={data.image}
                />
              ))}
            </div>
          )}
        </div>

        {classNames.length > 0 ? (
          <footer className="border-border flex flex-wrap gap-1 border-t px-4 py-2">
            {classNames.map((name) => (
              <Badge key={name} variant="secondary" className="text-[10px]">
                {name}
              </Badge>
            ))}
          </footer>
        ) : null}
      </div>
    </div>
  );
}
