// Friendly-name lookup for encoder aliases. The raw aliases (e.g.
// "finbert_fed_adjacent_xbank_dapt_retrieval") are how the backend
// registers checkpoints, but they read like internal slugs when
// rendered as captions on the workspace. Map the known aliases to a
// short human-readable label; fall back to a lightly cleaned version
// of the alias when nothing matches so unknown checkpoints stay
// readable rather than blank.

const FRIENDLY_NAMES: Record<string, string> = {
  finbert_fed_adjacent_xbank_dapt_retrieval: "FinBERT (Fed-adjacent + cross-bank DAPT)",
  finbert_fed_adjacent_xbank_dapt: "FinBERT (Fed-adjacent + cross-bank DAPT)",
  finbert_fed_adjacent: "FinBERT (Fed-adjacent)",
  finbert: "FinBERT",
  bge_large_en_v1_5: "BGE Large (en v1.5)",
  bge_large: "BGE Large",
  fomc_roberta: "FOMC-RoBERTa",
  roberta_base: "RoBERTa base",
  bert_base: "BERT base",
};

function prettifyAlias(alias: string): string {
  return alias
    .replace(/_/g, " ")
    .replace(/\bdapt\b/gi, "DAPT")
    .replace(/\bxbank\b/gi, "cross-bank")
    .replace(/\bfinbert\b/gi, "FinBERT")
    .replace(/\bbert\b/gi, "BERT")
    .replace(/\bbge\b/gi, "BGE")
    .replace(/\bfomc\b/gi, "FOMC")
    .replace(/\broberta\b/gi, "RoBERTa")
    .replace(/\s+/g, " ")
    .trim();
}

export function friendlyEncoderName(alias: string | null | undefined): string {
  if (!alias) return "—";
  const key = alias.toLowerCase();
  if (FRIENDLY_NAMES[key]) return FRIENDLY_NAMES[key];
  return prettifyAlias(alias);
}
