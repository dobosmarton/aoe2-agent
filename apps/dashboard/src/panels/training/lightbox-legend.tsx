import { classColor } from "@/panels/training/class-color";
import type { AnnotationDto } from "@/lib/training-api";

type LegendEntry = {
  readonly name: string;
  readonly classId: number;
  readonly count: number;
};

/** One entry per distinct class, carrying the id its colour derives from. */
const legendEntries = (annotations: readonly AnnotationDto[]): LegendEntry[] => {
  const byName = new Map<string, { classId: number; count: number }>();
  for (const ann of annotations) {
    const seen = byName.get(ann.class_name);
    if (seen === undefined) {
      byName.set(ann.class_name, { classId: ann.class_id, count: 1 });
    } else {
      seen.count += 1;
    }
  }
  return [...byName].map(([name, rest]) => ({ name, ...rest }));
};

/** Class chips below the image; pointing at one dims every other class's boxes.
 *
 * Takes the raw annotations and groups them itself, so the caller neither
 * imports the grouping helper nor decides when the legend is empty. */
export function LightboxLegend(props: {
  readonly annotations: readonly AnnotationDto[];
  readonly focus: string | null;
  readonly onFocusChange: (className: string | null) => void;
}): React.ReactElement | null {
  const { focus, onFocusChange } = props;
  const entries = legendEntries(props.annotations);

  if (entries.length === 0) {
    return null;
  }

  return (
    <footer className="border-border flex flex-wrap gap-1.5 border-t px-4 py-2">
      {entries.map((entry) => {
        const color = classColor(entry.classId);
        const active = focus === entry.name;
        return (
          // Hover *and* focus, so the highlight is reachable by keyboard.
          <button
            key={entry.name}
            type="button"
            className="flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[11px] transition-colors"
            style={{
              borderColor: color,
              color: active ? "black" : color,
              backgroundColor: active ? color : classColor(entry.classId, 0.12),
            }}
            onMouseEnter={() => {
              onFocusChange(entry.name);
            }}
            onMouseLeave={() => {
              onFocusChange(null);
            }}
            onFocus={() => {
              onFocusChange(entry.name);
            }}
            onBlur={() => {
              onFocusChange(null);
            }}
          >
            {entry.name}
            <span className="tabular-nums opacity-70">{entry.count}</span>
          </button>
        );
      })}
    </footer>
  );
}
