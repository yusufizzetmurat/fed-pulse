// Curated excerpts of real FOMC statements. The text is a faithful
// excerpt of the released statement; truncated for the sample picker
// to keep the workspace input from being overwhelmed. Dates correspond
// to the meeting (statement release date). ``symbol`` is the asset the
// loader should pre-select alongside the text and date — defaults to
// ^GSPC, the broadest equity index that maps naturally to FOMC events.
export interface SampleStatement {
  id: string;
  label: string;
  date: string;
  text: string;
  symbol: string;
}

export const SAMPLE_STATEMENTS: SampleStatement[] = [
  {
    id: "2024-09-18",
    label: "2024-09-18 (50bp cut)",
    date: "2024-09-18",
    symbol: "^GSPC",
    text:
      "Recent indicators suggest that economic activity has continued to expand at a solid pace. " +
      "Job gains have slowed, and the unemployment rate has moved up but remains low. Inflation " +
      "has made further progress toward the Committee's 2 percent objective but remains somewhat " +
      "elevated. The Committee has gained greater confidence that inflation is moving sustainably " +
      "toward 2 percent, and judges that the risks to achieving its employment and inflation goals " +
      "are roughly in balance. In light of the progress on inflation and the balance of risks, the " +
      "Committee decided to lower the target range for the federal funds rate by 1/2 percentage " +
      "point to 4-3/4 to 5 percent.",
  },
  {
    id: "2023-07-26",
    label: "2023-07-26 (25bp hike)",
    date: "2023-07-26",
    symbol: "^GSPC",
    text:
      "Recent indicators suggest that economic activity has been expanding at a moderate pace. " +
      "Job gains have been robust in recent months, and the unemployment rate has remained low. " +
      "Inflation remains elevated. The U.S. banking system is sound and resilient. Tighter credit " +
      "conditions for households and businesses are likely to weigh on economic activity, hiring, " +
      "and inflation. The extent of these effects remains uncertain. The Committee remains highly " +
      "attentive to inflation risks. In support of these goals, the Committee decided to raise the " +
      "target range for the federal funds rate to 5-1/4 to 5-1/2 percent.",
  },
  {
    id: "2022-06-15",
    label: "2022-06-15 (75bp hike)",
    date: "2022-06-15",
    symbol: "^GSPC",
    text:
      "Overall economic activity appears to have picked up after edging down in the first quarter. " +
      "Job gains have been robust in recent months, and the unemployment rate has remained low. " +
      "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, " +
      "higher energy prices, and broader price pressures. The invasion of Ukraine by Russia is " +
      "causing tremendous human and economic hardship. The Committee is strongly committed to " +
      "returning inflation to its 2 percent objective. The Committee decided to raise the target " +
      "range for the federal funds rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing " +
      "increases in the target range will be appropriate.",
  },
  {
    id: "2020-03-15",
    label: "2020-03-15 (emergency 100bp cut)",
    date: "2020-03-15",
    symbol: "^GSPC",
    text:
      "The coronavirus outbreak has harmed communities and disrupted economic activity in many " +
      "countries, including the United States. Global financial conditions have also been " +
      "significantly affected. Available economic data show that the U.S. economy came into this " +
      "challenging period on a strong footing. The effects of the coronavirus will weigh on " +
      "economic activity in the near term and pose risks to the economic outlook. In light of " +
      "these developments, the Committee decided to lower the target range for the federal funds " +
      "rate to 0 to 1/4 percent. The Committee expects to maintain this target range until it is " +
      "confident that the economy has weathered recent events and is on track to achieve its " +
      "maximum employment and price stability goals.",
  },
  {
    id: "2019-07-31",
    label: "2019-07-31 (25bp cut)",
    date: "2019-07-31",
    symbol: "^GSPC",
    text:
      "Information received since the Federal Open Market Committee met in June indicates that the " +
      "labor market remains strong and that economic activity has been rising at a moderate rate. " +
      "Job gains have been solid, on average, in recent months, and the unemployment rate has " +
      "remained low. Although growth of household spending has picked up from earlier in the year, " +
      "growth of business fixed investment has been soft. On a 12-month basis, overall inflation " +
      "and inflation for items other than food and energy are running below 2 percent. In light of " +
      "the implications of global developments for the economic outlook as well as muted inflation " +
      "pressures, the Committee decided to lower the target range for the federal funds rate to " +
      "2 to 2-1/4 percent.",
  },
];
