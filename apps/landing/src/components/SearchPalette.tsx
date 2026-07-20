import * as React from "react";
import { Command } from "cmdk";
import { Dialog, Heading, Modal, ModalOverlay } from "react-aria-components";
import { Search, X } from "lucide-react";

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
    // react-aria's ModalOverlay is both the portal and the backdrop, so the
    // Radix Portal/Overlay pair collapses into it; Modal carries what used to
    // be Dialog.Content. isDismissable restores Radix's click-outside-to-close
    // (Escape already closes by default).
    <ModalOverlay
      isOpen={open}
      onOpenChange={setOpen}
      isDismissable
      className="fixed inset-0 z-50 bg-black/60"
    >
      <Modal className="fixed left-1/2 top-[15%] z-50 w-[min(92vw,640px)] -translate-x-1/2 overflow-hidden rounded-lg border bg-popover shadow-2xl">
        <Dialog aria-describedby={undefined} className="outline-none">
          <Heading slot="title" className="sr-only">
            Search documentation
          </Heading>
          <Command shouldFilter={false} className="flex flex-col">
            <div className="flex items-center gap-2 border-b px-3 py-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <Command.Input
                autoFocus
                value={query}
                onValueChange={setQuery}
                placeholder="Search docs..."
                className="h-9 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                }}
                className="rounded-md p-1 hover:bg-accent"
                aria-label="Close"
              >
                <X className="h-4 w-4 text-muted-foreground" />
              </button>
            </div>
            <Command.List className="max-h-[60vh] overflow-y-auto p-2">
              {hits.length === 0 && query !== "" && (
                <Command.Empty className="px-3 py-6 text-center text-sm text-muted-foreground">
                  {pagefindReady
                    ? "No results."
                    : "Search index not built yet — run `pnpm build` to enable."}
                </Command.Empty>
              )}
              {hits.map((hit) => (
                <Command.Item
                  key={hit.url}
                  value={hit.url}
                  onSelect={() => {
                    window.location.href = hit.url;
                  }}
                  className="cursor-pointer rounded-md px-3 py-2 text-sm aria-selected:bg-accent aria-selected:text-accent-foreground"
                >
                  <p className="font-medium">{hit.title}</p>
                  <p
                    className="mt-0.5 line-clamp-2 text-xs text-muted-foreground"
                    dangerouslySetInnerHTML={{ __html: hit.excerpt }}
                  />
                </Command.Item>
              ))}
            </Command.List>
            <div className="border-t bg-muted/30 px-3 py-1.5 text-[10px] text-muted-foreground">
              <kbd className="rounded border bg-background px-1 py-0.5 font-mono">↵</kbd>{" "}
              open ·{" "}
              <kbd className="rounded border bg-background px-1 py-0.5 font-mono">esc</kbd>{" "}
              close ·{" "}
              <kbd className="rounded border bg-background px-1 py-0.5 font-mono">⌘K</kbd>{" "}
              toggle
            </div>
          </Command>
        </Dialog>
      </Modal>
    </ModalOverlay>
  );
}
