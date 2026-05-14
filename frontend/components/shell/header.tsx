import Link from "next/link";
import { Activity, BookOpen, Calendar, GitCompare, Github, History as HistoryIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

const NAV_ITEMS: Array<{ href: string; label: string; icon: React.ComponentType<{ className?: string }> }> = [
  { href: "/analyze", label: "Analyze", icon: Activity },
  { href: "/history", label: "History", icon: HistoryIcon },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/calendar", label: "Calendar", icon: Calendar },
];

export function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Activity className="h-4 w-4 text-primary" />
          <span>Fed Pulse</span>
          <Badge variant="outline" className="ml-1 text-[10px] uppercase tracking-wide">
            Research
          </Badge>
        </Link>
        <nav className="hidden items-center gap-1 sm:flex">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
            <Button key={href} asChild variant="ghost" size="sm">
              <Link href={href}>
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            </Button>
          ))}
        </nav>
        <div className="flex items-center gap-1">
          <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
            <Link href="https://github.com/yusufizzetmurat/fed-pulse/wiki" target="_blank" rel="noopener noreferrer">
              <BookOpen className="h-4 w-4" />
              Wiki
            </Link>
          </Button>
          <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
            <Link href="https://github.com/yusufizzetmurat/fed-pulse" target="_blank" rel="noopener noreferrer">
              <Github className="h-4 w-4" />
              Repo
            </Link>
          </Button>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
