import { useState } from "react";
import { BarChart3, ChevronLeft, Images, ScanEye } from "lucide-react";

import { Button } from "@/components/ui/button";
import { CoverageStats } from "@/panels/training/coverage-stats";
import { DatasetTable } from "@/panels/training/dataset-table";

type TrainingTab = "coverage" | "images";

const TABS: readonly { readonly id: TrainingTab; readonly label: string; readonly icon: React.ReactNode }[] = [
  { id: "coverage", label: "Coverage", icon: <BarChart3 className="size-4" /> },
  { id: "images", label: "Images", icon: <Images className="size-4" /> },
];

export function TrainingView(props: { readonly onExit: () => void }): React.ReactElement {
  const [tab, setTab] = useState<TrainingTab>("coverage");

  return (
    <div className="grid h-screen grid-cols-[240px_1fr]">
      <aside className="border-border bg-card flex flex-col border-r">
        <header className="border-border flex items-center gap-2 border-b px-4 py-3">
          <ScanEye className="text-primary size-4 shrink-0" />
          <div className="min-w-0">
            <h1 className="text-sm font-semibold leading-none">Detection Training</h1>
            <p className="text-muted-foreground mt-0.5 text-[11px]">Annotation tracker</p>
          </div>
        </header>
        <nav className="flex flex-col gap-1 p-2">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => {
                setTab(t.id);
              }}
              className={
                "flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors " +
                (t.id === tab
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50")
              }
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </nav>
        <div className="mt-auto p-2">
          <Button variant="ghost" size="sm" className="w-full justify-start" onClick={props.onExit}>
            <ChevronLeft className="size-4" />
            Back to Arena
          </Button>
        </div>
      </aside>

      <main className="bg-background flex min-h-0 min-w-0 flex-col">
        {tab === "coverage" ? <CoverageStats /> : <DatasetTable />}
      </main>
    </div>
  );
}
