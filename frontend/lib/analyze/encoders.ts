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
  finbert_fed_adjacent_xbank: "FinBERT (Fed-adjacent, cross-bank)",
  finbert_fed_adjacent: "FinBERT (Fed-adjacent)",
  finbert: "FinBERT",
  bge_large_en_v15: "BGE Large EN v1.5",
  bge_large_en_v1_5: "BGE Large EN v1.5",
  bge_large: "BGE Large",
  nomic_embed_text_v15: "Nomic Embed Text v1.5",
  nomic_embed_text_v1_5: "Nomic Embed Text v1.5",
  voyage_finance_2: "Voyage Finance 2",
  fomc_roberta: "FOMC-RoBERTa",
  roberta_base: "RoBERTa base",
  bert_base: "BERT base",
  bert_base_fed_adjacent: "BERT base (Fed-adjacent)",
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
