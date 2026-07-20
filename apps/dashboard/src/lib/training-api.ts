// Client for the training tracker API (apps/training-api on :8100, proxied in
// dev). Types mirror the frozen dataclass DTOs in apps/training-api/src/schemas.py.

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

const apiUrl = (path: string): string => {
  return `${API_BASE}${path}`;
};

/** Resolve a server-relative asset path (thumb_url / raw_url) against the base. */
export const trackerAssetUrl = (path: string): string => {
  return apiUrl(path);
};

export type ClassDto = {
  readonly id: number;
  readonly name: string;
}

export type ClassCoverageDto = {
  readonly class_id: number;
  readonly name: string;
  readonly real_instances: number;
  readonly synth_instances: number;
  readonly labeled_images: number;
}

export type CoverageDto = {
  readonly total_images: number;
  readonly labeled_images: number;
  readonly unlabeled_images: number;
  readonly classes: readonly ClassCoverageDto[];
  readonly zero_real_class_ids: readonly number[];
}

export type ImageRecordDto = {
  readonly id: number;
  readonly filename: string;
  readonly source: string;
  readonly width: number;
  readonly height: number;
  readonly thumb_url: string;
  readonly raw_url: string;
}

export type ImageListingDto = {
  readonly image: ImageRecordDto;
  readonly labeled: boolean;
  readonly annotation_count: number;
  readonly class_ids: readonly number[];
}

type AnnotationBase = {
  readonly id: number | null;
  readonly class_id: number;
  readonly class_name: string;
  readonly source: string;
  readonly status: string;
}

/** Discriminated on `geom_type` so `coords` narrows without a cast — the server
 * emits `[x, y, w, h]` for a bbox and `[[x, y], …]` for a polygon, both in
 * absolute pixels with a top-left origin (see geometry.py). */
export type AnnotationDto =
  | (AnnotationBase & {
      readonly geom_type: "bbox";
      readonly coords: readonly [number, number, number, number];
    })
  | (AnnotationBase & {
      readonly geom_type: "polygon";
      readonly coords: readonly (readonly [number, number])[];
    });

export type ImageDetailDto = {
  readonly image: ImageRecordDto;
  readonly annotations: readonly AnnotationDto[];
}

export type ImagePageDto = {
  readonly items: readonly ImageListingDto[];
  readonly total: number;
  readonly page: number;
  readonly page_size: number;
}

export type DatasetSummaryDto = {
  readonly id: number;
  readonly name: string;
  readonly created_at: string;
  readonly n_real_images: number;
  readonly n_synth_images: number;
  readonly notes: string | null;
}

/** null = show all; true/false filter by labeled state. */
export type LabeledFilter = boolean | null;

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const init = signal === undefined ? {} : { signal };
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    throw new Error(`GET ${path} failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function fetchClasses(signal?: AbortSignal): Promise<readonly ClassDto[]> {
  return getJson<readonly ClassDto[]>("/classes", signal);
}

export async function fetchStats(signal?: AbortSignal): Promise<CoverageDto> {
  return getJson<CoverageDto>("/stats", signal);
}

export async function fetchDatasets(signal?: AbortSignal): Promise<readonly DatasetSummaryDto[]> {
  return getJson<readonly DatasetSummaryDto[]>("/datasets", signal);
}

export async function fetchImageDetail(
  imageId: number,
  signal?: AbortSignal,
): Promise<ImageDetailDto> {
  return getJson<ImageDetailDto>(`/images/${String(imageId)}`, signal);
}

export async function fetchImages(
  labeled: LabeledFilter,
  page: number,
  signal?: AbortSignal,
): Promise<ImagePageDto> {
  const params = new URLSearchParams({ page: String(page) });
  if (labeled !== null) {
    params.set("labeled", String(labeled));
  }
  return getJson<ImagePageDto>(`/images?${params.toString()}`, signal);
}
