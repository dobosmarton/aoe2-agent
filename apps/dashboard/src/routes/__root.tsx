import type { QueryClient } from "@tanstack/react-query";
import { Outlet, createRootRouteWithContext } from "@tanstack/react-router";

/** Injected in main.tsx so route loaders can call
 * `context.queryClient.ensureQueryData(...)`. */
export type RouterContext = {
  readonly queryClient: QueryClient;
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: Outlet,
});
