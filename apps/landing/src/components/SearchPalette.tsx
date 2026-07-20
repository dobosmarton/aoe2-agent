import * as React from "react";

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

/**
 * Command-K search palette backed by Pagefind.
 *
 * Pagefind is generated *after* `astro build` runs (`pagefind --site dist`).
 * During `astro dev` the runtime doesn't exist, so a small loader script in
 * Site.astro attempts to fetch it and assigns it to `window.pagefind`. We
 * read from window here — that keeps Vite's static analyzer out of it
 * entirely (no `import(...)` of a non-existent file at build time).
 */

interface PagefindRuntime {
  search: (q: string) => Promise<{
    results: { id: string; data: () => Promise<PagefindResultData> }[];
  }>;
}

interface PagefindResultData {
  url: string;
  meta: { title?: string };
  excerpt: string;
}

declare global {
  interface Window {
    pagefind?: PagefindRuntime;
  }
}

interface Hit {
  url: string;
  title: string;
  excerpt: string;
}

export default function SearchPalette(): React.ReactElement {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  const [hits, setHits] = React.useState<Hit[]>([]);
  const [pagefindReady, setPagefindReady] = React.useState<boolean>(
    () => typeof window !== "undefined" && !!window.pagefind,
  );

  // Open on Cmd/Ctrl+K, and on click of the header search button.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    const onOpenSearch = (): void => setOpen(true);
    const headerBtn = document.getElementById("open-search");
    headerBtn?.addEventListener("click", onOpenSearch);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      headerBtn?.removeEventListener("click", onOpenSearch);
    };
  }, []);

  // Site.astro emits a "pagefind:ready" event after it successfully loads
  // the runtime from /pagefind/pagefind.js. If the runtime is already on
  // window when we mount, we skip waiting.
  React.useEffect(() => {
    if (pagefindReady) return;
    const onReady = (): void => setPagefindReady(true);
    window.addEventListener("pagefind:ready", onReady);
    return () => window.removeEventListener("pagefind:ready", onReady);
  }, [pagefindReady]);

  React.useEffect(() => {
    if (!query) {
      setHits([]);
      return;
    }
    const pf = window.pagefind;
    if (!pf) return;
    let cancelled = false;
    (async () => {
      const res = await pf.search(query);
      const top = await Promise.all(res.results.slice(0, 10).map((r) => r.data()));
      if (cancelled) return;
      setHits(
        top.map((d) => ({
          url: d.url,
          title: d.meta.title ?? d.url,
          excerpt: d.excerpt,
        })),
      );
    })();
    return () => {
      cancelled = true;
    };
  }, [query, pagefindReady]);

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Search documentation"
      description="Search the AoE2 LLM Arena docs"
      showCloseButton
    >
      {/* Results come from Pagefind, which has already ranked them, so the
          client-side filter is disabled — otherwise react-aria's Autocomplete
          would filter the server's hits a second time by input text. */}
      {/* Uncontrolled on purpose: Autocomplete owns the input's value through
          FieldInputContext, so also passing `inputValue` pins the field to our
          state and swallows keystrokes. We only observe the value to drive the
          Pagefind query. */}
      <Command filter={() => true} onInputChange={setQuery}>
        <CommandInput placeholder="Search docs..." />
        {/* The Menu must stay mounted even with no hits: Autocomplete wires the
            input to its collection via aria-controls, and unmounting the list
            breaks that link so the input stops accepting keystrokes. Empty and
            loading states go through renderEmptyState instead. */}
        <CommandList
          aria-label="Search results"
          items={hits}
          onAction={(key) => {
            window.location.href = String(key);
          }}
          renderEmptyState={() => (
            <CommandEmpty>
              {query === ""
                ? "Type to search the docs."
                : pagefindReady
                  ? "No results."
                  : "Search index not built yet — run `bun run build` to enable."}
            </CommandEmpty>
          )}
        >
          {(hit: Hit) => (
            <CommandItem id={hit.url} textValue={hit.title}>
              <div className="min-w-0">
                <p className="font-medium">{hit.title}</p>
                <p
                  className="mt-0.5 line-clamp-2 text-xs text-muted-foreground"
                  dangerouslySetInnerHTML={{ __html: hit.excerpt }}
                />
              </div>
            </CommandItem>
          )}
        </CommandList>
      </Command>
      <div className="border-t bg-muted/30 px-3 py-1.5 text-[10px] text-muted-foreground">
        <kbd className="rounded border bg-background px-1 py-0.5 font-mono">↵</kbd> open ·{" "}
        <kbd className="rounded border bg-background px-1 py-0.5 font-mono">esc</kbd> close ·{" "}
        <kbd className="rounded border bg-background px-1 py-0.5 font-mono">⌘K</kbd> toggle
      </div>
    </CommandDialog>
  );
}
