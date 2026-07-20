import type { QueryStatus } from "@tanstack/react-query";

/** The three-state loading vocabulary the presentational components speak.
 * Previously each data hook exported its own identical copy (RunsStatus,
 * SummariesStatus, …); now there is one, mapped from Query. */
export type LoadStatus = "loading" | "ready" | "error";

export function toLoadStatus(status: QueryStatus): LoadStatus {
  switch (status) {
    case "pending":
      return "loading";
    case "error":
      return "error";
    case "success":
      return "ready";
  }
}

/** Query exposes errors as `Error | null`; the components want a message. */
export function errorMessage(error: unknown): string | null {
  if (error === null || error === undefined) {
    return null;
  }
  return error instanceof Error ? error.message : String(error);
}
