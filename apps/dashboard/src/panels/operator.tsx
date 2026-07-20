import { useEffect, useReducer } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { GitBranch, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { EmptyState } from "@/components/empty-state";
import { AgeField } from "@/panels/age-field";
import { MutationSummary } from "@/panels/mutation-summary";
import { NumberField } from "@/panels/number-field";
import { OptionalNumberField } from "@/panels/optional-number-field";
import { RESOURCE_COLORS } from "@/panels/operator-fields";
import { createFork } from "@/lib/api";
import type { Age, MutationPatch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Form state machine
// ---------------------------------------------------------------------------

const SECTION_TITLE =
  "text-muted-foreground text-xs font-semibold uppercase tracking-wide";

type NumericField = Exclude<keyof MutationPatch, "age">;

interface FormState {
  parent_t: number;
  n_turns: number;
  reason: string;
  mutation: MutationPatch;
}

type FormAction =
  | { type: "set_parent_t"; value: number }
  | { type: "set_n_turns"; value: number }
  | { type: "set_reason"; value: string }
  | { type: "set_numeric"; field: NumericField; value: number | null }
  | { type: "set_age"; value: Age | null }
  | { type: "reset"; parent_t: number };

function reducer(state: FormState, action: FormAction): FormState {
  switch (action.type) {
    case "set_parent_t":
      return { ...state, parent_t: action.value };
    case "set_n_turns":
      return { ...state, n_turns: action.value };
    case "set_reason":
      return { ...state, reason: action.value };
    case "set_numeric": {
      const { [action.field]: _omit, ...rest } = state.mutation;
      const next: MutationPatch =
        action.value === null ? rest : { ...rest, [action.field]: action.value };
      return { ...state, mutation: next };
    }
    case "set_age": {
      const { age: _omit, ...rest } = state.mutation;
      const next: MutationPatch =
        action.value === null ? rest : { ...rest, age: action.value };
      return { ...state, mutation: next };
    }
    case "reset":
      return initialState(action.parent_t);
    default: {
      const _exhaustive: never = action;
      throw new Error(`unhandled action: ${String(_exhaustive)}`);
    }
  }
}

function initialState(parent_t: number): FormState {
  return { parent_t, n_turns: 10, reason: "", mutation: {} };
}

// ---------------------------------------------------------------------------
// Submit state
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface OperatorPanelProps {
  readonly currentRunId: string | null;
  readonly initialParentT: number | null;
}

export function OperatorPanel({
  currentRunId,
  initialParentT,
}: OperatorPanelProps): React.ReactElement {
  const [form, dispatch] = useReducer(reducer, initialState(initialParentT ?? 1));
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // Spawning a fork adds a run server-side, so the cached run list is stale the
  // moment it succeeds. Invalidating ["runs"] is what makes the child appear in
  // the sidebar without a page reload — the old code navigated to a run the
  // list did not yet know about.
  const fork = useMutation({
    mutationFn: createFork,
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ["runs"] });
      await navigate({
        to: "/runs/$runId",
        params: { runId: result.child_run_id },
        search: {},
      });
    },
  });

  // Reset the form when the user picks a different run.
  useEffect(() => {
    dispatch({ type: "reset", parent_t: initialParentT ?? 1 });
    fork.reset();
    // `fork` is a stable mutation object; depending on it would reset the form
    // on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentRunId, initialParentT]);

  if (currentRunId === null) {
    return (
      <EmptyState
        title="Select a run"
        hint="Pick a run from the sidebar to fork from."
      />
    );
  }

  function onSubmit(): void {
    if (currentRunId === null) {
      return;
    }
    fork.mutate({
      parent_run_id: currentRunId,
      parent_t: form.parent_t,
      mutation: form.mutation,
      n_turns: form.n_turns,
      reason: form.reason,
    });
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-3 overflow-auto p-4">
      <div className="flex items-center gap-2">
        <GitBranch className="text-event-fork size-4" />
        <h2 className="text-sm font-semibold">Spawn a fork</h2>
        <Badge variant="outline" className="ml-auto max-w-[60%] truncate font-mono text-xs">
          {currentRunId}
        </Badge>
      </div>

      <Card className="gap-3 py-4">
        <CardHeader className="px-4">
          <CardTitle className={SECTION_TITLE}>Fork point</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 px-4 sm:grid-cols-2">
          <NumberField
            label="Fork at turn"
            value={form.parent_t}
            min={0}
            onChange={(v) => dispatch({ type: "set_parent_t", value: v })}
          />
          <NumberField
            label="Replay N turns"
            value={form.n_turns}
            min={0}
            max={200}
            onChange={(v) => dispatch({ type: "set_n_turns", value: v })}
          />
          <div className="sm:col-span-2">
            <Label htmlFor="reason" className="text-muted-foreground text-xs">
              Reason (recorded on the world_mutation event)
            </Label>
            <Input
              id="reason"
              value={form.reason}
              onChange={(e) => dispatch({ type: "set_reason", value: e.target.value })}
              placeholder="e.g. starve them at age-up"
            />
          </div>
        </CardContent>
      </Card>

      <Card className="gap-3 py-4">
        <CardHeader className="px-4">
          <CardTitle className={SECTION_TITLE}>Mutation patch</CardTitle>
          <p className="text-muted-foreground text-xs">
            Leave a field blank to inherit the parent's value.
          </p>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 px-4 sm:grid-cols-2">
          {(["food", "wood", "gold", "stone", "population", "pop_cap"] as const).map(
            (field) => (
              <OptionalNumberField
                key={field}
                label={field}
                colorVar={RESOURCE_COLORS[field]}
                value={form.mutation[field] ?? null}
                onChange={(value) =>
                  dispatch({ type: "set_numeric", field, value })
                }
              />
            ),
          )}
          <AgeField
            value={form.mutation.age ?? null}
            onChange={(value) => dispatch({ type: "set_age", value })}
          />
        </CardContent>
      </Card>

      <Card className="gap-3 py-4">
        <CardHeader className="px-4">
          <CardTitle className={SECTION_TITLE}>Mutation summary</CardTitle>
        </CardHeader>
        <CardContent className="px-4">
          <MutationSummary mutation={form.mutation} />
        </CardContent>
      </Card>

      <div className="flex items-center justify-between gap-3">
        <Button
          isDisabled={fork.isPending}
          onClick={() => {
            void onSubmit();
          }}
        >
          {fork.isPending ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Spawning…
            </>
          ) : (
            <>
              <GitBranch className="size-4" />
              Spawn fork
            </>
          )}
        </Button>
        {fork.isError ? (
          <Badge variant="destructive" className="text-xs">
            {fork.error instanceof Error ? fork.error.message : String(fork.error)}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}
