import * as React from "react";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
  variant?: "card" | "inline";
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  variant = "card",
}: EmptyStateProps) {
  const body = (
    <div className={cn("flex flex-col items-center gap-2 text-center", className)}>
      {icon ? <div className="text-muted-foreground">{icon}</div> : null}
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description ? (
        <div className="max-w-md text-xs text-muted-foreground">{description}</div>
      ) : null}
      {action ? <div className="pt-1">{action}</div> : null}
    </div>
  );
  if (variant === "inline") {
    return <div className="py-8">{body}</div>;
  }
  return (
    <Card>
      <CardContent className="py-10">{body}</CardContent>
    </Card>
  );
}
