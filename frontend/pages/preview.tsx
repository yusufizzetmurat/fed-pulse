import * as React from "react";
import Head from "next/head";
import { ArrowUpRight, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

export default function PreviewPage() {
  return (
    <>
      <Head>
        <title>Design system — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main id="main-content" tabIndex={-1} className="container py-8 focus:outline-none">
          <div className="mb-8 space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">Design system</h1>
            <p className="max-w-2xl text-muted-foreground">
              Primitives the new dashboard will be built on. Tailwind + shadcn/ui with finance-focused
              tokens (hawkish / dovish / neutral). Theme toggle in the header switches every component
              at once.
            </p>
          </div>

          <Tabs defaultValue="primitives" className="space-y-6">
            <TabsList>
              <TabsTrigger value="primitives">Primitives</TabsTrigger>
              <TabsTrigger value="form">Form</TabsTrigger>
              <TabsTrigger value="states">States</TabsTrigger>
            </TabsList>

            <TabsContent value="primitives" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Buttons</CardTitle>
                  <CardDescription>Six variants, four sizes.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-wrap items-center gap-3">
                  <Button>Default</Button>
                  <Button variant="secondary">Secondary</Button>
                  <Button variant="outline">Outline</Button>
                  <Button variant="ghost">Ghost</Button>
                  <Button variant="destructive">Destructive</Button>
                  <Button variant="link">Link</Button>
                  <Button onClick={() => toast.success("Sentiment scored as hawkish")}>
                    <Sparkles /> Run analyze
                  </Button>
                  <Button variant="outline">
                    Open in wiki <ArrowUpRight />
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Stance badges</CardTitle>
                  <CardDescription>Three multi-axis output colours.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-wrap gap-3">
                  <Badge variant="hawkish">Hawkish · 0.62</Badge>
                  <Badge variant="dovish">Dovish · 0.18</Badge>
                  <Badge variant="neutral">Neutral · 0.20</Badge>
                  <Badge variant="outline">Source: FOMC statement</Badge>
                  <Badge>Factor +0.31</Badge>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="form" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Document ingestion</CardTitle>
                  <CardDescription>Three input modes will land in the next PR.</CardDescription>
                </CardHeader>
                <CardContent className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="symbol">Asset</Label>
                    <Select defaultValue="^GSPC">
                      <SelectTrigger id="symbol">
                        <SelectValue placeholder="Pick a benchmark" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="^GSPC">S&P 500</SelectItem>
                        <SelectItem value="^NDX">NASDAQ 100</SelectItem>
                        <SelectItem value="^DJI">Dow Jones</SelectItem>
                        <SelectItem value="^VIX">VIX</SelectItem>
                        <SelectItem value="GC=F">Gold</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="date">FOMC date</Label>
                    <Input id="date" type="date" defaultValue="2024-09-18" />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="text">FOMC text</Label>
                    <Textarea
                      id="text"
                      rows={5}
                      defaultValue="Recent indicators suggest economic activity has continued to expand at a solid pace."
                    />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="states" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Loading skeleton</CardTitle>
                  <CardDescription>Placeholder shape while async work resolves.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Skeleton className="h-4 w-2/3" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-32 w-full" />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </>
  );
}
