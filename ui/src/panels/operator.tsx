import { useEffect, useReducer, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { EmptyState } from "@/components/empty-state";
import { createFork } from "@/lib/api";
import type { Age, MutationPatch } from "@/lib/api";

// ---------------------------------------------------------------------------
// Form state machine
// ---------------------------------------------------------------------------

const _AGES: readonly Age[] = [
  "Dark Age",
  "Feudal Age",
  "Castle Age",
  "Imperial Age",
] as const;

const _MUTATION_FIELDS: ReadonlyArray<keyof MutationPatch> = [
  "food",
  "wood",
  "gold",
  "stone",
  "population",
  "pop_cap",
  "age",
] as const;

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

type SubmitState =
  | { kind: "idle" }
  | { kind: "submitting" }
  | { kind: "error"; message: string };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface OperatorPanelProps {
  readonly currentRunId: string | null;
  readonly initialParentT: number | null;
  readonly onOpenRun: (runId: string) => void;
}

export function OperatorPanel({
  currentRunId,
  initialParentT,
  onOpenRun,
}: OperatorPanelProps): React.ReactElement {
  const [form, dispatch] = useReducer(reducer, initialState(initialParentT ?? 1));
  const [submit, setSubmit] = useState<SubmitState>({ kind: "idle" });

  // Reset the form when the user picks a different run.
  useEffect(() => {
    dispatch({ type: "reset", parent_t: initialParentT ?? 1 });
    setSubmit({ kind: "idle" });
  }, [currentRunId, initialParentT]);

  if (currentRunId === null) {
    return (
      <EmptyState
        title="Select a run"
        hint="Pick a run from the sidebar to fork from."
      />
    );
  }

  async function onSubmit(): Promise<void> {
    if (currentRunId === null) {
      return;
    }
    setSubmit({ kind: "submitting" });
    try {
      const result = await createFork({
        parent_run_id: currentRunId,
        parent_t: form.parent_t,
        mutation: form.mutation,
        n_turns: form.n_turns,
        reason: form.reason,
      });
      setSubmit({ kind: "idle" });
      onOpenRun(result.child_run_id);
    } catch (error) {
      setSubmit({
        kind: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto p-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Fork target</CardTitle>
          <CardDescription className="font-mono text-xs">
            {currentRunId}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
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

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Mutation patch</CardTitle>
          <CardDescription className="text-xs">
            Leave a field blank to inherit the parent's value.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {_MUTATION_FIELDS.filter((field) => field !== "age").map((field) => (
            <OptionalNumberField
              key={field}
              label={field}
              value={form.mutation[field as NumericField] ?? null}
              onChange={(value) =>
                dispatch({ type: "set_numeric", field: field as NumericField, value })
              }
            />
          ))}
          <AgeField
            value={form.mutation.age ?? null}
            onChange={(value) => dispatch({ type: "set_age", value })}
          />
        </CardContent>
      </Card>

      <div className="flex items-center justify-between gap-3">
        <Button
          disabled={submit.kind === "submitting"}
          onClick={() => {
            void onSubmit();
          }}
        >
          {submit.kind === "submitting" ? "Spawning…" : "Spawn fork"}
        </Button>
        {submit.kind === "error" ? (
          <Badge variant="destructive" className="text-xs">
            {submit.message}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Field components
// ---------------------------------------------------------------------------

interface NumberFieldProps {
  readonly label: string;
  readonly value: number;
  readonly min?: number;
  readonly max?: number;
  readonly onChange: (value: number) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: NumberFieldProps): React.ReactElement {
  return (
    <div>
      <Label className="text-muted-foreground text-xs">{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const parsed = Number(e.target.value);
          if (Number.isFinite(parsed)) {
            onChange(parsed);
          }
        }}
      />
    </div>
  );
}

interface OptionalNumberFieldProps {
  readonly label: string;
  readonly value: number | null;
  readonly onChange: (value: number | null) => void;
}

function OptionalNumberField({
  label,
  value,
  onChange,
}: OptionalNumberFieldProps): React.ReactElement {
  return (
    <div>
      <Label className="text-muted-foreground text-xs capitalize">{label}</Label>
      <Input
        type="number"
        value={value ?? ""}
        placeholder="inherit"
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") {
            onChange(null);
            return;
          }
          const parsed = Number(raw);
          if (Number.isFinite(parsed)) {
            onChange(parsed);
          }
        }}
      />
    </div>
  );
}

interface AgeFieldProps {
  readonly value: Age | null;
  readonly onChange: (value: Age | null) => void;
}

function AgeField({ value, onChange }: AgeFieldProps): React.ReactElement {
  return (
    <div>
      <Label className="text-muted-foreground text-xs">age</Label>
      <select
        className="border-input bg-background text-foreground h-9 w-full rounded-md border px-3 text-sm shadow-xs focus-visible:outline-none"
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(raw === "" ? null : (raw as Age));
        }}
      >
        <option value="">inherit</option>
        {_AGES.map((a) => (
          <option key={a} value={a}>
            {a}
          </option>
        ))}
      </select>
    </div>
  );
}
