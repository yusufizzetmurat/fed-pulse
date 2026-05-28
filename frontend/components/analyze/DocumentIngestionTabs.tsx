import * as React from "react";
import axios from "axios";
import { FileText, Link as LinkIcon, Loader2, Type } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { resolveApiBaseUrl } from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";

interface DocumentIngestionTabsProps {
  text: string;
  onChange: (next: string) => void;
}

interface ParseResponse {
  text: string;
  char_count: number;
  source_kind: string;
  source_metadata: Record<string, string>;
}

export function DocumentIngestionTabs({ text, onChange }: DocumentIngestionTabsProps) {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [activeTab, setActiveTab] = React.useState<"paste" | "file" | "url">("paste");
  const [url, setUrl] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [lastSource, setLastSource] = React.useState<{ kind: string; meta: Record<string, string> } | null>(null);

  const submitParse = async (data: FormData) => {
    setBusy(true);
    try {
      const response = await axios.post<ParseResponse>(`${apiBaseUrl}/documents/parse`, data);
      onChange(response.data.text);
      setLastSource({ kind: response.data.source_kind, meta: response.data.source_metadata });
      toast.success(`Loaded ${response.data.char_count.toLocaleString()} chars from ${response.data.source_kind}`);
    } catch (err) {
      toast.error(errorMessage(err, "Failed to parse the document."));
    } finally {
      setBusy(false);
    }
  };

  const handleFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const data = new FormData();
    data.append("file", file);
    submitParse(data);
  };

  const handleUrl = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!url.trim()) return;
    const data = new FormData();
    data.append("url", url.trim());
    submitParse(data);
  };

  return (
    <div className="space-y-2">
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as typeof activeTab)}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <TabsList>
            <TabsTrigger value="paste">
              <Type className="h-3.5 w-3.5" /> Paste
            </TabsTrigger>
            <TabsTrigger value="file">
              <FileText className="h-3.5 w-3.5" /> PDF / DOCX
            </TabsTrigger>
            <TabsTrigger value="url">
              <LinkIcon className="h-3.5 w-3.5" /> URL
            </TabsTrigger>
          </TabsList>
          {lastSource ? (
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              source: {lastSource.kind}
              {lastSource.meta.page_count ? ` · ${lastSource.meta.page_count} pp` : ""}
              {lastSource.meta.paragraph_count ? ` · ${lastSource.meta.paragraph_count} para` : ""}
            </Badge>
          ) : null}
        </div>

        <TabsContent value="paste" className="space-y-2">
          <Label htmlFor="text">FOMC text</Label>
          <Textarea
            id="text"
            rows={8}
            required
            value={text}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Paste an FOMC statement excerpt…"
          />
        </TabsContent>

        <TabsContent value="file" className="space-y-2">
          <Label htmlFor="doc-file">Upload PDF or DOCX</Label>
          <Input
            id="doc-file"
            type="file"
            accept="application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.pdf,.docx"
            onChange={handleFile}
            disabled={busy}
          />
          {busy ? (
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" /> Parsing upload…
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Server uses pdfplumber / python-docx. Result populates the analyze text below.
            </p>
          )}
          {text ? (
            <Textarea readOnly rows={6} value={text} className="bg-muted/30" />
          ) : null}
        </TabsContent>

        <TabsContent value="url" className="space-y-2">
          <form onSubmit={handleUrl} className="space-y-2">
            <Label htmlFor="doc-url">Article URL</Label>
            <div className="flex gap-2">
              <Input
                id="doc-url"
                type="url"
                placeholder="https://www.federalreserve.gov/monetarypolicy/…"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                disabled={busy}
              />
              <Button type="submit" variant="outline" disabled={busy || !url.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Fetch"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Server fetches the page, strips chrome (nav / footer / scripts) and extracts the article body.
            </p>
          </form>
          {text ? (
            <Textarea readOnly rows={6} value={text} className="bg-muted/30" />
          ) : null}
        </TabsContent>
      </Tabs>
    </div>
  );
}
