import * as React from "react";
import { ArrowUpRight, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { KpiTile } from "@/components/ui/kpi-tile";
import { Skeleton } from "@/components/ui/skeleton";
import { Sparkline } from "@/components/ui/sparkline";

export function DesignSystemTab() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Buttons</CardTitle>
          <CardDescription>Six variants. Sizes adapt via the `size` prop.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2">
          <Button>Default</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="destructive">Destructive</Button>
          <Button variant="link">Link</Button>
          <Button onClick={() => toast.success("Sentiment scored as hawkish")}>
            <Sparkles className="h-3.5 w-3.5" />
            Run analyze
          </Button>
          <Button variant="outline">
            Open in wiki
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Stance + regime badges</CardTitle>
          <CardDescription>
            Hawkish / dovish / neutral palette doubles as the regime swatch (high / calm / normal).
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Badge variant="hawkish">Hawkish · 0.62</Badge>
          <Badge variant="dovish">Dovish · 0.18</Badge>
          <Badge variant="neutral">Neutral · 0.20</Badge>
          <Badge variant="outline">Source: FOMC statement</Badge>
          <Badge>Factor +0.31</Badge>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">KPI tile</CardTitle>
          <CardDescription>
            Label + mono value + delta chip + optional sparkline. Used across the workspace
            credibility row and the performance summary.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          <KpiTile
            label="Drift score"
            value={<span className="numeric">0.42</span>}
            sparkline={[0.31, 0.34, 0.36, 0.4, 0.42]}
            caption="vs prior 4 statements"
          />
          <KpiTile
            label="Argmax accuracy"
            value={<span className="numeric">61.2%</span>}
            delta={0.06}
            deltaFormatter={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}pp`}
            tone="up"
            caption="vs prior 30 runs"
          />
          <KpiTile
            label="Empirical coverage"
            value={<span className="numeric">76%</span>}
            delta={-0.04}
            deltaFormatter={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}pp`}
            tone="down"
            caption="80% nominal"
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Sparkline</CardTitle>
          <CardDescription>
            Recharts area with inferred up / down tone. Drops in any KPI cell.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          <Sparkline values={[1, 2, 3, 4, 5, 6]} />
          <Sparkline values={[6, 5, 4, 3, 2, 1]} />
          <Sparkline values={[3, 2, 4, 3, 4, 3]} tone="neutral" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Loading skeleton</CardTitle>
          <CardDescription>Placeholder shapes while async work resolves.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-4 w-2/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="h-32 w-full" />
        </CardContent>
      </Card>
    </div>
  );
}
