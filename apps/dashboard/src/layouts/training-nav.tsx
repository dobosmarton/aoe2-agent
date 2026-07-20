import { Link } from "@tanstack/react-router";
import { BarChart3, Images } from "lucide-react";

const TABS = [
  { to: "/training/coverage", label: "Coverage", icon: <BarChart3 className="size-4" /> },
  { to: "/training/images", label: "Images", icon: <Images className="size-4" /> },
] as const;

/** Section nav for the Training shell. */
export function TrainingNav(): React.ReactElement {
  return (
    <nav className="flex flex-col gap-1 p-2">
      {TABS.map((tab) => (
        // Link exposes the active state via render props, so this needs no
        // useState mirror of the current tab.
        <Link
          key={tab.to}
          to={tab.to}
          className="flex items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors"
          activeProps={{ className: "bg-accent text-accent-foreground font-medium" }}
          inactiveProps={{ className: "text-muted-foreground hover:bg-accent/50" }}
        >
          {tab.icon}
          {tab.label}
        </Link>
      ))}
    </nav>
  );
}
