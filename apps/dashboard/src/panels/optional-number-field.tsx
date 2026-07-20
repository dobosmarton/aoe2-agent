import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type OptionalNumberFieldProps = {
  readonly label: string;
  readonly value: number | null;
  readonly colorVar?: string | undefined;
  readonly onChange: (value: number | null) => void;
}

export function OptionalNumberField({
  label,
  value,
  colorVar,
  onChange,
}: OptionalNumberFieldProps): React.ReactElement {
  return (
    <div>
      <Label className="text-muted-foreground flex items-center gap-1.5 text-xs capitalize">
        {colorVar === undefined ? null : (
          <span
            className="inline-block size-2 rounded-[2px]"
            style={{ backgroundColor: colorVar }}
          />
        )}
        {label}
      </Label>
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
