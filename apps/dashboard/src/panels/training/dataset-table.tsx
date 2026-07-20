import { useQuery } from "@tanstack/react-query";
import { getRouteApi } from "@tanstack/react-router";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/load-status";
import { trackerImagesQueryOptions } from "@/lib/queries";
import { ImageLightbox } from "@/panels/training/image-lightbox";
import { trackerAssetUrl, type ImageListingDto, type LabeledFilter } from "@/lib/training-api";

// getRouteApi rather than importing the Route object: the route file already
// imports this component, so importing back would be a cycle.
const route = getRouteApi("/training/images");

const FILTERS: readonly { readonly label: string; readonly value: LabeledFilter }[] = [
  { label: "All", value: null },
  { label: "Labeled", value: true },
  { label: "Unlabeled", value: false },
];

function ImageCard(props: {
  readonly listing: ImageListingDto;
  readonly onOpen: (imageId: number) => void;
}): React.ReactElement {
  const { listing } = props;
  return (
    <figure className="border-border bg-card hover:border-primary focus-within:border-primary overflow-hidden rounded-md border transition-colors">
      {/* A real <button> rather than a click handler on the figure: keyboard
          focus and Enter/Space come for free, and the card stays a figure. */}
      <button
        type="button"
        title={`Open ${listing.image.filename}`}
        className="bg-muted flex aspect-video w-full cursor-pointer items-center justify-center overflow-hidden"
        onClick={() => {
          props.onOpen(listing.image.id);
        }}
      >
        <img
          src={trackerAssetUrl(listing.image.thumb_url)}
          alt={listing.image.filename}
          loading="lazy"
          className="h-full w-full object-cover"
        />
      </button>
      <figcaption className="flex items-center justify-between gap-2 px-2 py-1.5">
        <span className="text-muted-foreground truncate font-mono text-[11px]" title={listing.image.filename}>
          {listing.image.filename}
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

export function DatasetTable(): React.ReactElement {
  const { labeled, page, image } = route.useSearch();
  const navigate = route.useNavigate();
  const filter: LabeledFilter = labeled ?? null;
  const query = useQuery(trackerImagesQueryOptions(filter, page));

  // Search updaters build the next object explicitly rather than spreading and
  // deleting — omitting a key is how a param is cleared, and the project's
  // exactOptionalPropertyTypes forbids writing `key: undefined`.
  const selectFilter = (value: LabeledFilter): void => {
    void navigate({
      search: () => (value === null ? { page: 0 } : { labeled: value, page: 0 }),
      replace: true,
    });
  };
  const goToPage = (next: number): void => {
    void navigate({
      search: (prev) => ({
        ...(prev.labeled === undefined ? {} : { labeled: prev.labeled }),
        page: next,
      }),
    });
  };
  const setOpenImage = (next: number | null): void => {
    void navigate({
      search: (prev) => ({
        ...(prev.labeled === undefined ? {} : { labeled: prev.labeled }),
        page: prev.page,
        ...(next === null ? {} : { image: next }),
      }),
      replace: true,
    });
  };

  const data = query.data;
  const total = data?.total ?? 0;
  const pageSize = data?.page_size ?? 60;
  const lastPage = Math.max(0, Math.ceil(total / pageSize) - 1);

  return (
    <div className="flex min-h-0 flex-col">
      <div className="border-border flex items-center justify-between gap-3 border-b px-6 py-3">
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <Button
              key={f.label}
              size="sm"
              variant={f.value === filter ? "default" : "outline"}
              onClick={() => {
                selectFilter(f.value);
              }}
            >
              {f.label}
            </Button>
          ))}
        </div>
        <span className="text-muted-foreground font-mono text-xs tabular-nums">
          {total} images
        </span>
      </div>

      {query.isPending ? (
        <p className="text-muted-foreground p-6 text-sm">Loading images…</p>
      ) : query.isError || data === undefined ? (
        <p className="text-destructive p-6 text-sm">
          Failed to load images: {errorMessage(query.error) ?? "unknown error"}
        </p>
      ) : (
        <>
          {/* auto-rows-max + content-start: without them the row tracks are
              `auto`, and a grid with a definite height (flex-1) shrinks `auto`
              tracks to min-content when the rows don't fit — squashing every
              card to ~40px and clipping the thumbnail via overflow-hidden. */}
          <div className="grid min-h-0 flex-1 auto-rows-max content-start grid-cols-2 gap-3 overflow-auto p-6 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {data.items.map((listing) => (
              <ImageCard
                key={listing.image.id}
                listing={listing}
                onOpen={setOpenImage}
              />
            ))}
          </div>
          {lastPage > 0 ? (
            <div className="border-border flex items-center justify-center gap-3 border-t px-6 py-2">
              <Button
                size="sm"
                variant="outline"
                isDisabled={page === 0}
                onClick={() => {
                  goToPage(Math.max(0, page - 1));
                }}
              >
                Prev
              </Button>
              <span className="text-muted-foreground font-mono text-xs tabular-nums">
                {page + 1} / {lastPage + 1}
              </span>
              <Button
                size="sm"
                variant="outline"
                isDisabled={page >= lastPage}
                onClick={() => {
                  goToPage(Math.min(lastPage, page + 1));
                }}
              >
                Next
              </Button>
            </div>
          ) : null}
        </>
      )}

      <ImageLightbox
        imageId={image ?? null}
        onClose={() => {
          setOpenImage(null);
        }}
      />
    </div>
  );
}
