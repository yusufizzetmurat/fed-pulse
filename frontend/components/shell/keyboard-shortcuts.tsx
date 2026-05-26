import * as React from "react";

import { CommandPalette } from "@/components/shell/command-palette";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface ShortcutGroup {
  title: string;
  items: Array<{ keys: string[]; description: string }>;
}

const SHORTCUT_GROUPS: ShortcutGroup[] = [
  {
    title: "Navigation",
    items: [
      { keys: ["g", "w"], description: "Go to Workspace" },
      { keys: ["g", "h"], description: "Go to History" },
      { keys: ["g", "c"], description: "Go to Compare" },
      { keys: ["g", "k"], description: "Go to Calendar" },
      { keys: ["g", "p"], description: "Go to Performance" },
      { keys: ["g", "r"], description: "Go to Research" },
    ],
  },
  {
    title: "Command palette",
    items: [
      { keys: ["⌘ K"], description: "Open palette (jump-to-page, FOMC date, symbol)" },
      { keys: ["Ctrl K"], description: "Same as ⌘ K on non-Mac keyboards" },
    ],
  },
  {
    title: "Appearance",
    items: [{ keys: ["t"], description: "Toggle light / dark theme" }],
  },
  {
    title: "Help",
    items: [
      { keys: ["?"], description: "Open this shortcuts panel" },
      { keys: ["Esc"], description: "Close any open dialog" },
    ],
  },
];

const NAV_KEYS: Record<string, string> = {
  w: "/",
  h: "/history",
  c: "/compare",
  k: "/calendar",
  p: "/performance",
  r: "/research",
};

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if (target.isContentEditable) return true;
  return false;
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex min-w-[1.5rem] items-center justify-center rounded border border-border bg-muted px-1.5 py-0.5 text-[11px] font-mono font-medium text-foreground shadow-sm">
      {children}
    </kbd>
  );
}

/**
 * Mounts global keyboard shortcuts. `?` opens the bindings panel, `Cmd/Ctrl K`
 * opens the command palette, and two-key `g + x` sequences route between
 * top-level pages without going through the nav. Typing into inputs / textareas
 * is ignored so the shortcuts never fight a real keystroke.
 */
export function KeyboardShortcuts() {
  const [helpOpen, setHelpOpen] = React.useState(false);
  const [paletteOpen, setPaletteOpen] = React.useState(false);
  const gPendingRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const isCmd = event.metaKey || event.ctrlKey;
      if (isCmd && !event.altKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((value) => !value);
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isTypingTarget(event.target)) return;

      if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
        event.preventDefault();
        setHelpOpen((value) => !value);
        return;
      }

      if (event.key === "Escape" && helpOpen) {
        setHelpOpen(false);
        return;
      }

      if (event.key === "t" && !event.shiftKey) {
        const toggle = document.querySelector<HTMLButtonElement>(
          'button[aria-label$="theme"]',
        );
        if (toggle) {
          event.preventDefault();
          toggle.click();
        }
        return;
      }

      if (event.key === "g") {
        if (gPendingRef.current) window.clearTimeout(gPendingRef.current);
        gPendingRef.current = window.setTimeout(() => {
          gPendingRef.current = null;
        }, 1200);
        return;
      }

      if (gPendingRef.current && NAV_KEYS[event.key]) {
        event.preventDefault();
        window.clearTimeout(gPendingRef.current);
        gPendingRef.current = null;
        window.location.assign(NAV_KEYS[event.key]);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      if (gPendingRef.current) window.clearTimeout(gPendingRef.current);
    };
  }, [helpOpen]);

  return (
    <>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <Dialog open={helpOpen} onOpenChange={setHelpOpen}>
        <DialogContent aria-describedby="keyboard-shortcuts-description">
          <DialogHeader>
            <DialogTitle>Keyboard shortcuts</DialogTitle>
            <DialogDescription id="keyboard-shortcuts-description">
              Press <Kbd>?</Kbd> any time to reopen this panel. Shortcuts are ignored while typing
              into a text field.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            {SHORTCUT_GROUPS.map((group) => (
              <section key={group.title} aria-labelledby={`shortcut-group-${group.title}`}>
                <h2
                  id={`shortcut-group-${group.title}`}
                  className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {group.title}
                </h2>
                <ul className="space-y-1.5 text-sm">
                  {group.items.map((item) => (
                    <li
                      key={item.description}
                      className="flex items-center justify-between gap-3"
                    >
                      <span className="text-foreground">{item.description}</span>
                      <span className="flex items-center gap-1">
                        {item.keys.map((key, idx) => (
                          <React.Fragment key={`${item.description}-${idx}`}>
                            <Kbd>{key}</Kbd>
                            {idx < item.keys.length - 1 ? (
                              <span className="text-muted-foreground" aria-hidden="true">
                                then
                              </span>
                            ) : null}
                          </React.Fragment>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
