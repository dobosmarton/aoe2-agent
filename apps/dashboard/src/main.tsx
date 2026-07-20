import { StrictMode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { createRoot } from "react-dom/client";

import { routeTree } from "./routeTree.gen";
import "./index.css";

const queryClient = new QueryClient();

const router = createRouter({
  routeTree,
  context: { queryClient },
  // Query owns caching; without this the router would layer its own preload
  // staleness on top and the two would disagree about when data is fresh.
  defaultPreloadStaleTime: 0,
  Wrap: ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  ),
});

// Makes every <Link to="..."> and useSearch() call typed against this app's
// actual route tree.
declare module "@tanstack/react-router" {
  // Must stay an `interface`: this merges into the router's own Register
  // declaration, and a `type` alias cannot merge — it collapses every
  // Link/useSearch/useParams type in the app back to `any`.
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions
  interface Register {
    router: typeof router;
  }
}

const rootEl = document.getElementById("root");
if (rootEl === null) {
  throw new Error("Root element #root missing from index.html");
}

createRoot(rootEl).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
