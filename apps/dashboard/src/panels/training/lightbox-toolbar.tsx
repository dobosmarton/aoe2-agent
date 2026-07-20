import { Maximize2, Minus, Plus, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { MAX_ZOOM, MIN_ZOOM } from "@/panels/training/use-zoom-pan";
import type { ImageDetailDto } from "@/lib/training-api";

/** Filename, dimensions and zoom controls across the top of the lightbox. */
export function LightboxToolbar(props: {
  /** null until the image detail query resolves. */
  readonly detail: ImageDetailDto | null;
  readonly zoom: number;
  readonly onZoomBy: (factor: number) => void;
  readonly onReset: () => void;
  readonly onClose: () => void;
}): React.ReactElement {
  const { detail, zoom, onZoomBy, onReset, onClose } = props;

  return (
    <header className="border-border flex items-center gap-3 border-b px-4 py-2">
      <span className="truncate font-mono text-xs" title={detail?.image.filename}>
        {detail?.image.filename ?? "Loading…"}
      </span>
      {detail !== null ? (
        <span className="text-muted-foreground shrink-0 font-mono text-[11px] tabular-nums">
          {detail.image.width}×{detail.image.height} · {detail.annotations.length} boxes
        </span>
      ) : null}
      <div className="ml-auto flex shrink-0 items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          aria-label="Zoom out"
          isDisabled={zoom <= MIN_ZOOM}
          onClick={() => {
            onZoomBy(1 / 1.3);
          }}
        >
          <Minus className="size-4" />
        </Button>
        <span className="text-muted-foreground w-12 text-center font-mono text-[11px] tabular-nums">
          {Math.round(zoom * 100)}%
        </span>
        <Button
          variant="ghost"
          size="sm"
          aria-label="Zoom in"
          isDisabled={zoom >= MAX_ZOOM}
          onClick={() => {
            onZoomBy(1.3);
          }}
        >
          <Plus className="size-4" />
        </Button>
        <Button variant="ghost" size="sm" aria-label="Reset zoom" onClick={onReset}>
          <Maximize2 className="size-4" />
        </Button>
        <Button variant="ghost" size="sm" aria-label="Close" onClick={onClose}>
          <X className="size-4" />
        </Button>
      </div>
    </header>
  );
}
