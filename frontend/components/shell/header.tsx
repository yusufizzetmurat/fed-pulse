import Link from "next/link";
import {
  Activity,
  BookOpen,
  Calendar,
  Cpu,
  FlaskConical,
  GitCompare,
  Github,
  Gavel,
  History as HistoryIcon,
  LineChart,
} from "lucide-react";

import { SkipLink } from "@/components/shell/skip-link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

const NAV_ITEMS: Array<{ href: string; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { href: "/analyze", label: "Analyze", icon: Activity },
  { href: "/research", label: "Research", icon: FlaskConical },
  { href: "/training", label: "Training", icon: Cpu },
  { href: "/decisions", label: "Decisions", icon: Gavel },
  { href: "/history", label: "History", icon: HistoryIcon },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/performance", label: "Performance", icon: LineChart },
  { href: "/calendar", label: "Calendar", icon: Calendar },
];

export function Header() {
  return (
    <>
      <SkipLink />
      <header className="sticky top-0 z-40 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-between gap-4">
          <Link
            href="/"
            className="flex items-center gap-2 rounded-sm font-semibold tracking-tight focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
            aria-label="Fed Pulse home"
          >
            <Activity className="h-4 w-4 text-primary" aria-hidden="true" />
            <span>Fed Pulse</span>
            <Badge variant="outline" className="ml-1 text-[10px] uppercase tracking-wide">
              Research
            </Badge>
          </Link>
          <nav aria-label="Primary" className="hidden items-center gap-1 sm:flex">
            {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
              <Button key={href} asChild variant="ghost" size="sm">
                <Link href={href}>
                  <Icon className="h-4 w-4" aria-hidden="true" />
                  {label}
                </Link>
              </Button>
            ))}
          </nav>
          <div className="flex items-center gap-1">
            <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
              <Link
                href="https://github.com/yusufizzetmurat/fed-pulse/wiki"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Open the project wiki in a new tab"
              >
                <BookOpen className="h-4 w-4" aria-hidden="true" />
                Wiki
              </Link>
            </Button>
            <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
              <Link
                href="https://github.com/yusufizzetmurat/fed-pulse"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Open the GitHub repository in a new tab"
              >
                <Github className="h-4 w-4" aria-hidden="true" />
                Repo
              </Link>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>
    </>
  );
}
