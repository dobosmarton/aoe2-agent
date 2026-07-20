import type { QueryStatus } from "@tanstack/react-query";

/** The three-state loading vocabulary the presentational components speak.
 * Previously each data hook exported its own identical copy (RunsStatus,
 * SummariesStatus, …); now there is one, mapped from Query. */
export type LoadStatus = "loading" | "ready" | "error";

export const toLoadStatus = (status: QueryStatus): LoadStatus => {
  switch (status) {
    case "pending":
      return "loading";
    case "error":
      return "error";
    case "success":
      return "ready";
  }
};

/** Query exposes errors as `Error | null`; the components want a message.
 *
 * A rejected value need not be an Error — a thrown plain object would stringify
 * to a useless "[object Object]", so those are serialised instead. */
export const errorMessage = (error: unknown): string | null => {
  if (error === null || error === undefined) {
    return null;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  // Covers thrown objects, numbers and booleans; returns undefined for values
  // JSON cannot represent (symbol, function), hence the fallback.
  return JSON.stringify(error) ?? "Unknown error";
};
