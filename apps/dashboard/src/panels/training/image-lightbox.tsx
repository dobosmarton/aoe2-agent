import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { QueryFallback } from "@/components/query-fallback";
import { trackerImageDetailQueryOptions } from "@/lib/queries";
import { trackerAssetUrl } from "@/lib/training-api";
import { AnnotationBox } from "@/panels/training/annotation-box";
import { LightboxLegend } from "@/panels/training/lightbox-legend";
import { LightboxToolbar } from "@/panels/training/lightbox-toolbar";
import { useZoomPan } from "@/panels/training/use-zoom-pan";

/**
 * Full-screen image detail with its annotation overlay.
 *
 * Mounted only while an image is open, and keyed by image id by the caller, so
 * switching images remounts this and zoom/focus start fresh — no reset effect.
 */
export function ImageLightbox(props: {
  readonly imageId: number;
  readonly onClose: () => void;
}): React.ReactElement {
  const { imageId, onClose } = props;
  const query = useQuery(trackerImageDetailQueryOptions(imageId));
  const detail = query.data ?? null;

  const viewport = useRef<HTMLDivElement | null>(null);
  const content = useRef<HTMLDivElement | null>(null);
  const { zoom, offset, reset, zoomBy, onWheel, onPointerDown } = useZoomPan(
    viewport,
    content,
  );
  const [focus, setFocus] = useState<string | null>(null);

  // Attached by hand because React registers `wheel` passively at the root,
  // where preventDefault is a no-op.
  useEffect(() => {
    const node = viewport.current;
    if (node === null) {
      return;
    }
    node.addEventListener("wheel", onWheel, { passive: false });
    return (): void => {
      node.removeEventListener("wheel", onWheel);
    };
  }, [onWheel]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        onClose();
      } else if (event.key === "+" || event.key === "=") {
        zoomBy(1.3);
      } else if (event.key === "-" || event.key === "_") {
        zoomBy(1 / 1.3);
      } else if (event.key === "0") {
        reset();
      }
    };
    window.addEventListener("keydown", onKey);
    return (): void => {
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose, reset, zoomBy]);

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
        <LightboxToolbar
          detail={detail}
          zoom={zoom}
          onZoomBy={zoomBy}
          onReset={reset}
          onClose={onClose}
        />

        <div
          ref={viewport}
          className="flex min-h-0 flex-1 items-center justify-center overflow-hidden p-4"
          style={{ cursor: zoom > 1 ? "grab" : "default", touchAction: "none" }}
          onPointerDown={onPointerDown}
          onDoubleClick={() => {
            if (zoom > 1) {
              reset();
            } else {
              zoomBy(2.5);
            }
          }}
        >
          {detail === null ? (
            <QueryFallback noun="image" query={query} />
          ) : (
            // Shrink-wraps the image so the percentage-positioned boxes align;
            // scaling this wrapper zooms image and overlay as one unit.
            <div
              ref={content}
              className="relative inline-block"
              style={{
                transform: `translate(${String(offset.x)}px, ${String(offset.y)}px) scale(${String(zoom)})`,
                transformOrigin: "0 0",
                // No `will-change: transform`: that hint makes Chrome rasterise
                // the layer once and stretch the bitmap, which blurs the labels.
              }}
            >
              <img
                src={trackerAssetUrl(detail.image.raw_url)}
                alt={detail.image.filename}
                draggable={false}
                className="block max-h-[70vh] max-w-full select-none object-contain"
              />
              {detail.annotations.map((ann) => (
                <AnnotationBox
                  key={`${String(ann.class_id)}:${ann.coords.toString()}`}
                  annotation={ann}
                  record={detail.image}
                  zoom={zoom}
                  focus={focus}
                />
              ))}
            </div>
          )}
        </div>

        <LightboxLegend
          annotations={detail?.annotations ?? []}
          focus={focus}
          onFocusChange={setFocus}
        />
      </div>
    </div>
  );
}
