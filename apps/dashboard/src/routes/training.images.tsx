import { createFileRoute } from "@tanstack/react-router";

import { trackerImagesQueryOptions } from "@/lib/queries";
import { DatasetTable } from "@/panels/training/dataset-table";
import type { LabeledFilter } from "@/lib/training-api";

export type ImagesSearch = {
  /** Absent = show all; true/false filter by labeled state. */
  readonly labeled?: boolean;
  readonly page: number;
  /** Image id open in the lightbox, if any. */
  readonly image?: number;
}

export const Route = createFileRoute("/training/images")({
  validateSearch: (search: Record<string, unknown>): ImagesSearch => {
    const page = Number(search.page);
    const image = Number(search.image);
    // Keys are omitted rather than set to undefined (exactOptionalPropertyTypes).
    return {
      page: Number.isFinite(page) && page >= 0 ? page : 0,
      ...(search.labeled === true || search.labeled === "true"
        ? { labeled: true }
        : search.labeled === false || search.labeled === "false"
          ? { labeled: false }
          : {}),
      ...(Number.isFinite(image) && image > 0 ? { image } : {}),
    };
  },
  loaderDeps: ({ search }) => ({ labeled: search.labeled, page: search.page }),
  loader: ({ context, deps }) =>
    context.queryClient.ensureQueryData(
      trackerImagesQueryOptions(deps.labeled ?? (null as LabeledFilter), deps.page),
    ),
  component: DatasetTable,
});
