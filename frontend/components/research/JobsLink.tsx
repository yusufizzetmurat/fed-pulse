import Link from "next/link";
import { ArrowUpRight, Cpu } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function JobsLink() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Cpu className="h-4 w-4 text-primary" />
          Training jobs
        </CardTitle>
        <CardDescription>
          Legacy in-process train queue, kept for inspecting the old runtime-adaptation runs.
          The current production path is checkpoint-only and runs on the workspace; this
          surface stays available for archival and debugging.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Button asChild variant="outline" size="sm">
          <Link href="/training">
            Open jobs queue
            <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}
