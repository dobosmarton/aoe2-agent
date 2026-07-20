import { Label } from "@/components/ui/label";
import { AGES } from "@/panels/operator-fields";
import type { Age } from "@/lib/api";

type AgeFieldProps = {
  readonly value: Age | null;
  readonly onChange: (value: Age | null) => void;
}

export function AgeField({ value, onChange }: AgeFieldProps): React.ReactElement {
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
        {AGES.map((a) => (
          <option key={a} value={a}>
            {a}
          </option>
        ))}
      </select>
    </div>
  );
}
