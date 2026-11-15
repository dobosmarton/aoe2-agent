import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, RotateCcw } from "lucide-react";

import { cn } from "@/lib/utils";
import { ArtifactFrame } from "@/components/ui/artifact-frame";

// One iteration of a repeating pipeline, ported from the author's blog, drawn as a
// cycle: each phase is a card with its numbered steps, and arrows lead from card to card
// and back to the start. Four phases go around a 2×2 grid (clockwise); phones get one
// column that ends with a "back to the start" line. A background step runs alongside
// the loop, not inside it. Static: no client directive.

type Step = { label: string; time?: string; background?: boolean; note?: string };

// `time` covers the phase as a whole, for steps that are only measured together.
type Phase = { label: string; time?: string; steps: Step[] };

type LoopStepsProps = {
  title: string;
  caption?: string;
  phases: Phase[];
  // The label on the arrow that closes the loop, e.g. "next frame".
  loopLabel?: string;
};

const arrowClass = "size-4 shrink-0 text-muted-foreground";

// Clockwise around a 2×2 grid with arrow cells in between: top left, top right, bottom
// right, bottom left.
const cyclePlacement = [
  "sm:col-start-1 sm:row-start-1",
  "sm:col-start-3 sm:row-start-1",
  "sm:col-start-3 sm:row-start-3",
  "sm:col-start-1 sm:row-start-3",
];

function PhaseCard({ phase, firstNumber, className }: { phase: Phase; firstNumber: number; className?: string }) {
  return (
    <section className={cn("overflow-hidden rounded-md border border-foreground/20", className)}>
      <header className="flex items-baseline justify-between gap-2 border-b border-foreground/15 bg-foreground/[0.03] px-3 py-1.5">
        <p className="font-mono text-[11px] tracking-wider text-muted-foreground uppercase">[ {phase.label} ]</p>
        {phase.time && (
          <p className="font-mono text-[11px] whitespace-nowrap text-muted-foreground tabular-nums">{phase.time}</p>
        )}
      </header>
      <ol className="list-none space-y-1.5 px-3 py-2">
        {phase.steps.map((step, i) => (
          <li key={step.label} className="grid grid-cols-[1.25rem_1fr_auto] items-baseline gap-x-2">
            <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
              {String(firstNumber + i).padStart(2, "0")}
            </span>
            <div className="min-w-0">
              <p className="text-sm leading-5 text-foreground">
                {step.label}
                {step.background && (
                  <span className="ml-1.5 font-mono text-[10px] tracking-wider whitespace-nowrap text-muted-foreground uppercase">
                    [ async ]
                  </span>
                )}
              </p>
              {step.note && <p className="text-xs leading-snug text-muted-foreground">{step.note}</p>}
            </div>
            <span className="font-mono text-xs whitespace-nowrap text-muted-foreground tabular-nums">{step.time}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function LoopSteps({ title, caption, phases, loopLabel = "next turn" }: LoopStepsProps) {
  // The number of the first step in each phase: steps are numbered across phases.
  const firstNumber = phases.map((_, i) => 1 + phases.slice(0, i).reduce((n, phase) => n + phase.steps.length, 0));
  const total = phases.reduce((count, phase) => count + phase.steps.length, 0);
  const cycle = phases.length === 4;

  return (
    <ArtifactFrame label={title} meta={`${total} steps`} caption={caption}>
      <div
        className={cn(
          "grid grid-cols-1 gap-y-1.5 font-sans",
          cycle && "sm:grid-cols-[minmax(0,1fr)_1.75rem_minmax(0,1fr)] sm:grid-rows-[auto_2.25rem_auto] sm:gap-y-0",
        )}
        role="group"
        aria-label={title}
      >
        {phases.map((phase, i) => (
          <div key={phase.label} className="contents">
            <PhaseCard phase={phase} firstNumber={firstNumber[i]!} className={cycle ? cyclePlacement[i] : undefined} />
            {/* Phones: a down arrow after every card but the last. */}
            {i < phases.length - 1 && (
              <ArrowDown aria-hidden className={cn(arrowClass, "justify-self-center", cycle && "sm:hidden")} />
            )}
          </div>
        ))}
        {cycle && (
          <>
            {/* Desktop: the arrows between the cards, clockwise. */}
            <ArrowRight
              aria-hidden
              className={cn(arrowClass, "hidden self-center justify-self-center sm:col-start-2 sm:row-start-1 sm:block")}
            />
            <ArrowDown
              aria-hidden
              className={cn(arrowClass, "hidden self-center justify-self-center sm:col-start-3 sm:row-start-2 sm:block")}
            />
            <ArrowLeft
              aria-hidden
              className={cn(arrowClass, "hidden self-center justify-self-center sm:col-start-2 sm:row-start-3 sm:block")}
            />
            <p className="hidden items-center justify-center gap-1.5 self-center font-mono text-[11px] text-muted-foreground sm:col-start-1 sm:row-start-2 sm:flex">
              <ArrowUp aria-hidden className={arrowClass} />
              {loopLabel}
            </p>
          </>
        )}
        {/* Phones (or a non-cycle layout): the loop closes with a line back to step 1. */}
        <p
          className={cn(
            "flex items-center justify-center gap-1.5 pt-1 font-mono text-[11px] text-muted-foreground",
            cycle && "sm:hidden",
          )}
        >
          <RotateCcw aria-hidden className={arrowClass} />
          back to step 01, {loopLabel}
        </p>
      </div>
    </ArtifactFrame>
  );
}
