import Link from "next/link";
import { ArrowUpRight, Gavel } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function DecisionsLink() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Gavel className="h-4 w-4 text-primary" />
          Next-FOMC rate decision
        </CardTitle>
        <CardDescription>
          Ordinal-logit forecast over rate-decision classes (cut 50 → hike 75) with the
          OIS-implied baseline alongside, plus the walk-forward CV history and feature-family
          attribution table. Reads <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">data/artifacts/next_fomc/</code>.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Button asChild variant="outline" size="sm">
          <Link href="/decisions">
            Open decisions surface
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}
