import type { RunSummary } from "@/lib/events";

// Backend origin. Empty string = same-origin (prod build served by FastAPI,
// or dev with Vite proxy). Set VITE_API_BASE_URL=http://host:port to point
// the dev SPA at a remote/non-proxied backend.
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export async function fetchRuns(signal?: AbortSignal): Promise<readonly RunSummary[]> {
  const init = signal === undefined ? {} : { signal };
  const response = await fetch(apiUrl("/runs"), init);
  if (!response.ok) {
    throw new Error(`GET /runs failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as readonly RunSummary[];
}

export function eventsUrl(runId: string): string {
  return apiUrl(`/events?run_id=${encodeURIComponent(runId)}`);
}

// ---------------------------------------------------------------------------
// POST /forks
// ---------------------------------------------------------------------------

export type Age = "Dark Age" | "Feudal Age" | "Castle Age" | "Imperial Age";

export type MutationPatch = Partial<{
  food: number;
  wood: number;
  gold: number;
  stone: number;
  population: number;
  pop_cap: number;
  age: Age;
}>;

export interface ForkSpec {
  readonly parent_run_id: string;
  readonly parent_t: number;
  readonly mutation: MutationPatch;
  readonly n_turns: number;
  readonly reason: string;
}

export interface ForkResult {
  readonly child_run_id: string;
  readonly db_path: string;
  readonly profile_used: string;
}

export async function createFork(spec: ForkSpec): Promise<ForkResult> {
  const response = await fetch(apiUrl("/forks"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!response.ok) {
    // FastAPI returns either {detail: string} (HTTPException) or a
    // {detail: [{...}]} array (Pydantic validation). Surface whichever
    // we can parse, falling back to the HTTP status line.
    const body = (await response.json().catch(() => null)) as
      | { detail?: unknown }
      | null;
    const detail =
      body && typeof body.detail === "string"
        ? body.detail
        : `POST /forks failed: ${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return (await response.json()) as ForkResult;
}
