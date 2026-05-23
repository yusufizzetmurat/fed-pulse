import type {
  CredibilityResponse,
  MultiAxisResponse,
  XaiResponse,
} from "./types";

export const SAMPLE_MULTI_AXIS: MultiAxisResponse = {
  stance: {
    label: "hawkish",
    confidence: 0.62,
    distribution: { hawkish: 0.62, neutral: 0.21, dovish: 0.17 },
  },
  factor: {
    value: 0.31,
    confidence: 0.74,
    range: [0.14, 0.49],
  },
  certainty: {
    label: "certain",
    confidence: 0.68,
    distribution: { certain: 0.68, uncertain: 0.18, neutral: 0.14 },
  },
  topic: {
    label: "macro",
    primary: "macro",
    confidence: 0.58,
    secondary: ["forward_guidance", "market_reaction"],
    distribution: { macro: 0.58, forward_guidance: 0.27, market_reaction: 0.1, other: 0.05 },
  },
};

export const SAMPLE_XAI: XaiResponse = {
  method: "integrated_gradients",
  sentences: [
    {
      text:
        "Recent indicators suggest economic activity has continued to expand at a solid pace.",
      score: 0.18,
      topTokens: [
        { token: "solid", weight: 0.42 },
        { token: "expand", weight: 0.31 },
        { token: "pace", weight: 0.18 },
        { token: "continued", weight: 0.12 },
        { token: "activity", weight: 0.09 },
      ],
    },
    {
      text:
        "Inflation has eased over the past year but remains elevated.",
      score: 0.58,
      topTokens: [
        { token: "elevated", weight: 0.61 },
        { token: "inflation", weight: 0.48 },
        { token: "remains", weight: 0.27 },
        { token: "eased", weight: -0.21 },
        { token: "past", weight: 0.08 },
      ],
    },
    {
      text:
        "The Committee is strongly committed to returning inflation to its 2 percent objective.",
      score: 0.81,
      topTokens: [
        { token: "strongly", weight: 0.72 },
        { token: "committed", weight: 0.66 },
        { token: "returning", weight: 0.41 },
        { token: "2", weight: 0.19 },
        { token: "objective", weight: 0.14 },
      ],
    },
    {
      text:
        "In considering any adjustments, the Committee will carefully assess incoming data.",
      score: -0.12,
      topTokens: [
        { token: "carefully", weight: -0.32 },
        { token: "assess", weight: -0.18 },
        { token: "incoming", weight: -0.07 },
        { token: "considering", weight: -0.05 },
        { token: "adjustments", weight: 0.03 },
      ],
    },
  ],
};

export const SAMPLE_CREDIBILITY: CredibilityResponse = {
  drift_score: 0.34,
  drift_trend: [0.21, 0.28, 0.31, 0.34],
  realized_vs_stated_gap: -0.08,
  market_implied_gap: 0.12,
  months_since_reversal: 14,
};
