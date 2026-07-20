/** Branding block at the top of a sidebar: icon, title, subtitle, optional action. */
export function SidebarHeader(props: {
  readonly icon: React.ReactNode;
  readonly title: string;
  readonly subtitle: string;
  /** Rendered flush right — a nav button in the Arena shell, absent in Training. */
  readonly action?: React.ReactNode;
}): React.ReactElement {
  return (
    <header className="border-border flex items-center gap-2 border-b px-4 py-3">
      {props.icon}
      <div className="min-w-0 flex-1">
        <h1 className="text-sm font-semibold leading-none">{props.title}</h1>
        <p className="text-muted-foreground mt-0.5 text-[11px]">{props.subtitle}</p>
      </div>
      {props.action}
    </header>
  );
}
