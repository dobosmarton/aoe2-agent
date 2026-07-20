import { Badge } from "@/components/ui/badge";
import { trackerAssetUrl, type ImageListingDto } from "@/lib/training-api";

/** Thumbnail tile in the dataset grid; opens the lightbox when activated. */
export function ImageCard(props: {
  readonly listing: ImageListingDto;
  readonly onOpen: (imageId: number) => void;
}): React.ReactElement {
  const { listing, onOpen } = props;
  const { filename, thumb_url, id } = listing.image;

  return (
    <figure className="border-border bg-card hover:border-primary focus-within:border-primary overflow-hidden rounded-md border transition-colors">
      {/* A real <button> rather than a click handler on the figure: keyboard
          focus and Enter/Space come for free, and the card stays a figure. */}
      <button
        type="button"
        title={`Open ${filename}`}
        className="bg-muted flex aspect-video w-full cursor-pointer items-center justify-center overflow-hidden"
        onClick={() => {
          onOpen(id);
        }}
      >
        <img
          src={trackerAssetUrl(thumb_url)}
          alt={filename}
          loading="lazy"
          className="h-full w-full object-cover"
        />
      </button>
      <figcaption className="flex items-center justify-between gap-2 px-2 py-1.5">
        <span
          className="text-muted-foreground truncate font-mono text-[11px]"
          title={filename}
        >
          {filename}
        </span>
        {listing.labeled ? (
          <Badge variant="secondary" className="shrink-0 text-[10px]">
            {listing.annotation_count} boxes
          </Badge>
        ) : (
          <Badge variant="outline" className="text-muted-foreground shrink-0 text-[10px]">
            unlabeled
          </Badge>
        )}
      </figcaption>
    </figure>
  );
}
