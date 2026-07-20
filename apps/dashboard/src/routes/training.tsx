import { Link, Outlet, createFileRoute, useNavigate } from "@tanstack/react-router";
import { BarChart3, ChevronLeft, Images, ScanEye } from "lucide-react";

import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/training")({
  component: TrainingLayout,
});

const TABS = [
  { to: "/training/coverage", label: "Coverage", icon: <BarChart3 className="size-4" /> },
  { to: "/training/images", label: "Images", icon: <Images className="size-4" /> },
] as const;

function TrainingLayout(): React.ReactElement {
  const navigate = useNavigate();

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
            // Link exposes the active state via render props, so the section
            // nav no longer needs its own useState mirror of the current tab.
            <Link
              key={t.to}
              to={t.to}
              className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors"
              activeProps={{ className: "bg-accent text-accent-foreground font-medium" }}
              inactiveProps={{ className: "text-muted-foreground hover:bg-accent/50" }}
            >
              {t.icon}
              {t.label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto p-2">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start"
            onClick={() => {
              void navigate({ to: "/runs" });
            }}
          >
            <ChevronLeft className="size-4" />
            Back to Arena
          </Button>
        </div>
      </aside>

      <main className="bg-background flex min-h-0 min-w-0 flex-col">
        <Outlet />
      </main>
    </div>
  );
}
