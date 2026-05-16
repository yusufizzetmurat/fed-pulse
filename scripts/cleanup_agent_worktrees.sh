#!/usr/bin/env bash
# Prune locked / orphaned ``.claude/worktrees/agent-*`` directories.
#
# Background: agent sessions started with ``Agent({isolation: "worktree"})``
# materialise a git worktree under ``.claude/worktrees/agent-<id>/`` and
# do not always tear them down cleanly. The directories themselves are
# git-ignored so they never bloat the repo, but they accumulate on
# disk and confuse ``git worktree list`` when the underlying refs are
# gone.
#
# What this script does:
#
# 1. Runs ``git worktree prune`` to drop registry entries whose
#    on-disk directory has already been deleted.
# 2. Scans ``.claude/worktrees/agent-*/`` and removes any directory
#    that no longer has a corresponding ``git worktree list`` entry.
# 3. Skips the directory the script itself is running from, so
#    invoking it from inside an agent worktree does not delete that
#    worktree out from under the running process.
#
# This script is read-only with respect to active worktrees: an entry
# that ``git worktree list`` still surfaces is left alone. Run it from
# the repo root.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${REPO_ROOT}" ]; then
  echo "cleanup_agent_worktrees: not inside a git work tree" >&2
  exit 1
fi

cd "${REPO_ROOT}"

WORKTREE_DIR=".claude/worktrees"
if [ ! -d "${WORKTREE_DIR}" ]; then
  echo "cleanup_agent_worktrees: no ${WORKTREE_DIR} directory; nothing to do"
  exit 0
fi

# Step 1: ask git to drop stale registry entries first.
git worktree prune --verbose || true

# Step 2: collect the set of active worktree paths git still tracks.
ACTIVE_PATHS="$(
  git worktree list --porcelain |
    awk '$1 == "worktree" { print $2 }'
)"

# Resolve to absolute paths for the membership test.
RUNNING_FROM="$(pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

is_active() {
  local candidate="$1"
  while IFS= read -r active; do
    if [ "${active}" = "${candidate}" ]; then
      return 0
    fi
  done <<< "${ACTIVE_PATHS}"
  return 1
}

removed=0
for entry in "${WORKTREE_DIR}"/agent-*; do
  [ -d "${entry}" ] || continue
  abs="$(cd -- "${entry}" && pwd -P)"
  # Never remove the worktree currently running this script.
  case "${RUNNING_FROM}" in
    "${abs}"|"${abs}"/*) continue ;;
  esac
  case "${SCRIPT_DIR}" in
    "${abs}"|"${abs}"/*) continue ;;
  esac
  if is_active "${abs}"; then
    continue
  fi
  echo "removing orphaned worktree: ${entry}"
  rm -rf -- "${entry}"
  removed=$((removed + 1))
done

echo "cleanup_agent_worktrees: removed ${removed} orphan(s)"
