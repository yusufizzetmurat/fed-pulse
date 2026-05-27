import * as React from "react";
import Head from "next/head";
import { Settings as SettingsIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { toast } from "sonner";

import { AssetPicker } from "@/components/analyze/AssetPicker";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchSettingsCheckpoints, resolveApiBaseUrl } from "@/lib/analyze/api";
import type { Horizon, SettingsCheckpoint } from "@/lib/analyze/types";
import {
  DEFAULT_HORIZON,
  DEFAULT_SYMBOL,
  HORIZON_VALUES,
  loadWorkspacePrefs,
  saveWorkspacePrefs,
} from "@/lib/workspace-prefs";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatModified(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function roleLabel(role: string): string {
  switch (role) {
    case "forecaster":
      return "Forecaster";
    case "multi_axis":
      return "Multi-axis classifier";
    case "lora_adapter":
      return "LoRA adapter";
    case "calibration":
      return "Calibration";
    default:
      return role.replace(/_/g, " ");
  }
}

function CheckpointRow({ checkpoint }: { checkpoint: SettingsCheckpoint }) {
  // #342: render the inference-contract surface. Legacy checkpoints
  // (no sidecar) carry ``inference_contract_status === "sidecar_absent"``
  // and render with a neutral "legacy" badge. Post-#341 checkpoints
  // carry one badge per declared kwarg; the badge goes red when the
  // serving wiring does not supply that kwarg, neutral otherwise.
  const contractStatus = checkpoint.inference_contract_status ?? null;
  const requiredKwargs = checkpoint.required_kwargs ?? [];
  const supplied = checkpoint.supplied_at_inference ?? {};

  return (
    <li
      className="flex flex-col gap-2 rounded-md border border-border bg-background/50 p-3 sm:flex-row sm:items-start sm:justify-between"
      aria-label={checkpoint.filename}
    >
      <div className="flex items-start gap-3">
        <span
          className={`mt-1 inline-block h-2 w-2 rounded-full ${
            checkpoint.is_active ? "bg-up" : "bg-muted-foreground/40"
          }`}
          aria-hidden="true"
        />
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="numeric text-sm font-medium">{checkpoint.filename}</span>
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              {roleLabel(checkpoint.role)}
            </Badge>
            {checkpoint.is_active ? (
              <Badge variant="dovish" className="text-[10px]">
                active
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px] text-muted-foreground">
                inactive
              </Badge>
            )}
            {checkpoint.output_mode ? (
              <Badge variant="outline" className="text-[10px] capitalize">
                {checkpoint.output_mode}
              </Badge>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            <span className="numeric">{formatBytes(checkpoint.size_bytes)}</span>
            <span>·</span>
            <span className="numeric">{formatModified(checkpoint.modified_at)}</span>
            {checkpoint.encoder_alias ? (
              <>
                <span>·</span>
                <span className="numeric">{checkpoint.encoder_alias}</span>
              </>
            ) : null}
            {checkpoint.role === "forecaster" ? (
              <>
                <span>·</span>
                <span>
                  conformal sidecar:{" "}
                  {checkpoint.conformal_sidecar_present ? (
                    <span className="text-up">present</span>
                  ) : (
                    <span className="text-down">missing</span>
                  )}
                </span>
              </>
            ) : null}
          </div>
          {checkpoint.role === "forecaster" && contractStatus ? (
            <div
              className="flex flex-wrap items-center gap-1.5 pt-1"
              aria-label="inference contract kwargs"
            >
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                inference contract:
              </span>
              {contractStatus === "sidecar_absent" ? (
                <Badge
                  variant="outline"
                  className="text-[10px] text-muted-foreground"
                  data-testid="contract-legacy-badge"
                >
                  legacy
                </Badge>
              ) : requiredKwargs.length === 0 ? (
                <Badge variant="outline" className="text-[10px] text-muted-foreground">
                  no required kwargs
                </Badge>
              ) : (
                requiredKwargs.map((name) => {
                  const isSupplied = supplied[name] === true;
                  return (
                    <Badge
                      key={name}
                      variant={isSupplied ? "outline" : "hawkish"}
                      className="text-[10px] numeric"
                      data-testid={
                        isSupplied
                          ? `contract-kwarg-ok-${name}`
                          : `contract-kwarg-missing-${name}`
                      }
                      title={
                        isSupplied
                          ? `${name} is supplied by the serving wiring`
                          : `${name} declared by the checkpoint sidecar but NOT supplied by the serving wiring`
                      }
                    >
                      {name}
                    </Badge>
                  );
                })
              )}
            </div>
          ) : null}
        </div>
      </div>
    </li>
  );
}

function ModelsSection() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<SettingsCheckpoint[]>([]);
  const [modelsDir, setModelsDir] = React.useState<string>("");
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetchSettingsCheckpoints(apiBaseUrl, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setData(response.checkpoints);
        setModelsDir(response.models_dir);
        setError(null);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError((err as Error).message || "Could not load checkpoints.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [apiBaseUrl]);

  const grouped = React.useMemo(() => {
    const buckets = new Map<string, SettingsCheckpoint[]>();
    for (const cp of data) {
      const role = cp.role || "other";
      const list = buckets.get(role) ?? [];
      list.push(cp);
      buckets.set(role, list);
    }
    return [...buckets.entries()];
  }, [data]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Models</CardTitle>
        <CardDescription>
          {modelsDir ? (
            <>
              Read-only inventory of <code className="rounded bg-muted px-1 font-mono text-xs">{modelsDir}</code>.
              Active flag points at the file each service is currently loaded from. Live swap is intentionally
              not wired — drop a new checkpoint into the directory and restart the backend container to switch.
            </>
          ) : (
            "Read-only inventory of the backend models directory."
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : error ? (
          <EmptyState
            variant="inline"
            title="Could not load checkpoints"
            description={<p>{error}</p>}
          />
        ) : data.length === 0 ? (
          <EmptyState
            variant="inline"
            title="No checkpoints on disk"
            description="The backend models directory is empty. Train a checkpoint and drop it into the path above."
          />
        ) : (
          <div className="space-y-4">
            {grouped.map(([role, list]) => (
              <section key={role} className="space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  {roleLabel(role)}
                </h3>
                <ul className="space-y-2">
                  {list.map((cp) => (
                    <CheckpointRow key={cp.filename} checkpoint={cp} />
                  ))}
                </ul>
              </section>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function WorkspacePrefsSection() {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [defaultSymbol, setDefaultSymbol] = React.useState<string>(DEFAULT_SYMBOL);
  const [defaultHorizon, setDefaultHorizon] = React.useState<Horizon>(DEFAULT_HORIZON);

  // Hydrate from localStorage after mount so SSR + client agree.
  React.useEffect(() => {
    const prefs = loadWorkspacePrefs();
    setDefaultSymbol(prefs.defaultSymbol);
    setDefaultHorizon(prefs.defaultHorizon);
  }, []);

  const persist = React.useCallback(
    (symbol: string, horizon: Horizon) => {
      saveWorkspacePrefs({ defaultSymbol: symbol, defaultHorizon: horizon });
    },
    [],
  );

  const handleSymbolChange = (next: string) => {
    setDefaultSymbol(next);
    persist(next, defaultHorizon);
    toast.success(`Default symbol saved as ${next}`);
  };

  const handleHorizonChange = (next: Horizon) => {
    setDefaultHorizon(next);
    persist(defaultSymbol, next);
    toast.success(`Default horizon saved as ${next}`);
  };

  const handleReset = () => {
    setDefaultSymbol(DEFAULT_SYMBOL);
    setDefaultHorizon(DEFAULT_HORIZON);
    persist(DEFAULT_SYMBOL, DEFAULT_HORIZON);
    toast.success("Workspace defaults reset");
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Workspace defaults</CardTitle>
        <CardDescription>
          Persisted in this browser. The workspace reads them on first load and any analyze you fire on
          the home page uses these as the starting symbol / horizon.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="settings-symbol">Default symbol</Label>
            <AssetPicker id="settings-symbol" value={defaultSymbol} onChange={handleSymbolChange} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="settings-horizon">Default horizon</Label>
            <Select value={defaultHorizon} onValueChange={(value) => handleHorizonChange(value as Horizon)}>
              <SelectTrigger id="settings-horizon">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {HORIZON_VALUES.map((horizon) => (
                  <SelectItem key={horizon} value={horizon}>
                    {horizon}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="settings-theme">Theme</Label>
            <Select
              value={theme ?? resolvedTheme ?? "system"}
              onValueChange={(value) => setTheme(value)}
            >
              <SelectTrigger id="settings-theme">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dark">Dark</SelectItem>
                <SelectItem value="light">Light</SelectItem>
                <SelectItem value="system">Follow system</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="mt-4">
          <Button variant="outline" size="sm" onClick={handleReset}>
            Reset to factory defaults
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function SettingsPage() {
  return (
    <>
      <Head>
        <title>Settings — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-5 py-6 focus:outline-none">
          <div className="space-y-1">
            <h1 className="flex items-center gap-2">
              <SettingsIcon className="h-6 w-6 text-primary" />
              Settings
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Read-only inventory of the model files the backend has loaded, plus the per-browser
              defaults the workspace uses when you open it.
            </p>
          </div>

          <ModelsSection />
          <WorkspacePrefsSection />
        </main>
      </div>
    </>
  );
}
