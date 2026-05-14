import { CredibilityPanel } from "@/components/analyze/CredibilityPanel";
import { MultiAxisCards } from "@/components/analyze/MultiAxisCards";
import { XaiPanel } from "@/components/analyze/XaiPanel";
import {
  SAMPLE_CREDIBILITY,
  SAMPLE_MULTI_AXIS,
  SAMPLE_XAI,
} from "@/lib/analyze/fixtures";
import type {
  CredibilityResponse,
  MultiAxisResponse,
  XaiResponse,
} from "@/lib/analyze/types";

interface PreviewPanelsProps {
  multiAxis?: MultiAxisResponse;
  xai?: XaiResponse;
  credibility?: CredibilityResponse;
  slot: "cards" | "xai" | "credibility";
}

export default function PreviewPanels({ multiAxis, xai, credibility, slot }: PreviewPanelsProps) {
  if (slot === "cards") {
    return <MultiAxisCards multiAxis={multiAxis ?? SAMPLE_MULTI_AXIS} previewMode={!multiAxis} />;
  }
  if (slot === "xai") {
    return <XaiPanel xai={xai ?? SAMPLE_XAI} previewMode={!xai} />;
  }
  return (
    <CredibilityPanel
      credibility={credibility ?? SAMPLE_CREDIBILITY}
      previewMode={!credibility}
    />
  );
}
