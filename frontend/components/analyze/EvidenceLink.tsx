import { BookOpen } from "lucide-react";

import { cn } from "@/lib/utils";

// Anchor on the rendered wiki page. GitHub slugifies the markdown
// heading "6.7 Post-correction programme + Path B result (2026-05-23)"
// into "67-post-correction-programme--path-b-result-2026-05-23" by
// stripping punctuation and lower-casing words; the section numbers
// without dots collapse into the front of the slug. To keep the
// component honest we link the wiki page itself and let the small
// section-tag chip carry the §6.x string for the reader.
const WIKI_BASE = "https://github.com/yusufizzetmurat/fed-pulse/wiki/06-Deep-Learning-Roadmap";

export interface EvidenceLinkProps {
  // §-prefixed section id like "6.7" or "6.15". Rendered verbatim
  // inside the chip; the reader knows the convention from the wiki.
  section: string;
  // Short one-liner pinning the panel's claim to the section.
  label: string;
  // Optional extra anchor suffix (e.g. "#three-way-comparison") when
  // a sub-heading on the same wiki page is the load-bearing cell. The
  // slug is the caller's responsibility — keep it copy-pasted from the
  // GitHub-rendered table-of-contents so future heading renames flag
  // the breakage as a 404 rather than a silent miss.
  anchor?: string;
  className?: string;
}

export function EvidenceLink({ section, label, anchor, className }: EvidenceLinkProps) {
  const href = anchor ? `${WIKI_BASE}${anchor}` : WIKI_BASE;
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-dashed border-border bg-muted/30 px-2 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground",
        className,
      )}
      title={`Evidence: §${section} — ${label}`}
    >
      <BookOpen className="h-3 w-3" aria-hidden="true" />
      <span className="numeric">evidence · §{section}</span>
      <span className="hidden sm:inline">· {label}</span>
    </a>
  );
}
