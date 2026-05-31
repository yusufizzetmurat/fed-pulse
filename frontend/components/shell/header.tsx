import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  Activity,
  BookOpen,
  Calendar,
  FlaskConical,
  GitCompare,
  Github,
  History as HistoryIcon,
  LineChart,
  Menu,
  Settings as SettingsIcon,
  Terminal,
  X,
} from "lucide-react";

import { SkipLink } from "@/components/shell/skip-link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const NAV_ITEMS: Array<{ href: string; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { href: "/", label: "Workspace", icon: Activity },
  { href: "/console", label: "Terminal", icon: Terminal },
  { href: "/decisions", label: "Predictions", icon: LineChart },
  { href: "/history", label: "History", icon: HistoryIcon },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/calendar", label: "Calendar", icon: Calendar },
  { href: "/performance", label: "Performance", icon: LineChart },
  { href: "/research", label: "Research", icon: FlaskConical },
];

function isActive(currentPath: string, href: string): boolean {
  if (href === "/") return currentPath === "/";
  return currentPath === href || currentPath.startsWith(`${href}/`);
}

export function Header() {
  const router = useRouter();
  const currentPath = (router.asPath || "/").split("?")[0];
  const [mobileOpen, setMobileOpen] = React.useState(false);

  // Auto-close the mobile menu on route change so a tap on a nav item
  // doesn't leave the panel hanging over the new page.
  React.useEffect(() => {
    const events = router.events;
    if (!events) return;
    const handler = () => setMobileOpen(false);
    events.on("routeChangeStart", handler);
    return () => events.off("routeChangeStart", handler);
  }, [router.events]);

  return (
    <>
      <SkipLink />
      <header className="sticky top-0 z-40 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-12 items-center justify-between gap-4">
          <Link
            href="/"
            className="flex items-center gap-2 rounded-sm font-semibold tracking-tight focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            aria-label="Fed Pulse workspace"
          >
            <Activity className="h-4 w-4 text-primary" aria-hidden="true" />
            <span>Fed Pulse</span>
            <Badge
              variant="outline"
              className="ml-1 hidden text-[10px] uppercase tracking-wide sm:inline-flex"
            >
              Volatility Regime
            </Badge>
          </Link>
          <nav aria-label="Primary" className="hidden items-center gap-0.5 sm:flex">
            {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
              const active = isActive(currentPath, href);
              return (
                <Button
                  key={href}
                  asChild
                  variant={active ? "secondary" : "ghost"}
                  size="sm"
                  className={cn("h-7 px-2", active && "bg-secondary")}
                >
                  <Link href={href} aria-current={active ? "page" : undefined}>
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                    {label}
                  </Link>
                </Button>
              );
            })}
          </nav>
          <div className="flex items-center gap-0.5">
            <Button asChild variant="ghost" size="sm" className="hidden h-7 px-2 sm:inline-flex">
              <Link
                href="https://github.com/yusufizzetmurat/fed-pulse/wiki"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Open the project wiki in a new tab"
              >
                <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
                Wiki
              </Link>
            </Button>
            <Button asChild variant="ghost" size="sm" className="hidden h-7 px-2 sm:inline-flex">
              <Link
                href="https://github.com/yusufizzetmurat/fed-pulse"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Open the GitHub repository in a new tab"
              >
                <Github className="h-3.5 w-3.5" aria-hidden="true" />
                Repo
              </Link>
            </Button>
            <Button
              asChild
              variant={isActive(currentPath, "/settings") ? "secondary" : "ghost"}
              size="icon"
              className="h-7 w-7"
            >
              <Link href="/settings" aria-label="Open settings">
                <SettingsIcon className="h-3.5 w-3.5" aria-hidden="true" />
              </Link>
            </Button>
            <ThemeToggle />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="min-h-[44px] min-w-[44px] sm:hidden"
              aria-label={mobileOpen ? "Close navigation menu" : "Open navigation menu"}
              aria-expanded={mobileOpen}
              aria-controls="mobile-nav-panel"
              onClick={() => setMobileOpen((open) => !open)}
            >
              {mobileOpen ? (
                <X className="h-4 w-4" aria-hidden="true" />
              ) : (
                <Menu className="h-4 w-4" aria-hidden="true" />
              )}
            </Button>
          </div>
        </div>
        {mobileOpen ? (
          <nav
            id="mobile-nav-panel"
            aria-label="Primary mobile"
            className="border-t border-border bg-background sm:hidden"
          >
            <ul className="container flex flex-col gap-1 py-3">
              {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
                const active = isActive(currentPath, href);
                return (
                  <li key={href}>
                    <Link
                      href={href}
                      aria-current={active ? "page" : undefined}
                      className={cn(
                        "flex min-h-[44px] items-center gap-3 rounded-md px-3 text-sm font-medium",
                        active
                          ? "bg-secondary text-foreground"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      )}
                    >
                      <Icon className="h-4 w-4" aria-hidden="true" />
                      {label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        ) : null}
      </header>
    </>
  );
}
