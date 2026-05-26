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
  Settings as SettingsIcon,
} from "lucide-react";

import { SkipLink } from "@/components/shell/skip-link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const NAV_ITEMS: Array<{ href: string; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { href: "/", label: "Workspace", icon: Activity },
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
            <Badge variant="outline" className="ml-1 text-[10px] uppercase tracking-wide">
              vol-regime
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
          </div>
        </div>
      </header>
    </>
  );
}
