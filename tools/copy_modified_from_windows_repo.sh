#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${1:-/mnt/c/Users/z00595wr/work/strands_solver_py}"
DEST_REPO="${2:-$PWD}"
WINDOWS_GIT_EXE="/mnt/c/Program Files/Git/cmd/git.exe"

if [[ ! -d "$SOURCE_REPO/.git" ]]; then
  echo "error: source repo not found or not a git repo: $SOURCE_REPO" >&2
  exit 1
fi

if [[ ! -d "$DEST_REPO" ]]; then
  echo "error: destination directory not found: $DEST_REPO" >&2
  exit 1
fi

# Choose git implementation for source status detection:
# - For repos on /mnt/c/*, prefer Windows git.exe to match Windows line-ending/index behavior.
# - Otherwise, use the current Linux git.
if [[ "$SOURCE_REPO" == /mnt/c/* ]] && [[ -x "$WINDOWS_GIT_EXE" ]]; then
  SOURCE_GIT=("$WINDOWS_GIT_EXE")
  if command -v wslpath >/dev/null 2>&1; then
    SOURCE_REPO_ARG="$(wslpath -w "$SOURCE_REPO")"
  else
    SOURCE_REPO_ARG="$SOURCE_REPO"
  fi
  echo "Using Windows git for source detection: $WINDOWS_GIT_EXE"
else
  SOURCE_GIT=(git)
  SOURCE_REPO_ARG="$SOURCE_REPO"
  echo "Using Linux git for source detection: git"
fi

# Collect tracked files that are modified (staged and/or unstaged),
# limited to src/ and tests/ to mirror `git status` "modified:" entries.
mapfile -t modified_paths < <(
  {
    "${SOURCE_GIT[@]}" -C "$SOURCE_REPO_ARG" diff --name-only --diff-filter=M -- src tests
    "${SOURCE_GIT[@]}" -C "$SOURCE_REPO_ARG" diff --name-only --cached --diff-filter=M -- src tests
  } | sort -u
)

if [[ ${#modified_paths[@]} -eq 0 ]]; then
  echo "No tracked modified files found under src/ or tests/ in: $SOURCE_REPO"
  exit 0
fi

copied=0
skipped_missing=0
normalized_lf=0

for rel_path in "${modified_paths[@]}"; do
  # Keep only paths that are explicitly under src/ or tests/
  if [[ "$rel_path" != src/* && "$rel_path" != tests/* ]]; then
    continue
  fi

  src_file="$SOURCE_REPO/$rel_path"
  dst_file="$DEST_REPO/$rel_path"

  if [[ ! -f "$src_file" ]]; then
    echo "skip (missing in source): $rel_path"
    ((skipped_missing+=1))
    continue
  fi

  mkdir -p "$(dirname "$dst_file")"
  cp -f "$src_file" "$dst_file"

  # Normalize Windows line endings (CRLF) to Linux line endings (LF).
  # This keeps local diffs clean when source files come from Windows checkout.
  if command -v perl >/dev/null 2>&1; then
    perl -i -pe 's/\r$//' "$dst_file"
  else
    # Fallback without perl.
    sed -i 's/\r$//' "$dst_file"
  fi

  echo "copied: $rel_path"
  ((copied+=1))
  ((normalized_lf+=1))
done

echo "Done. copied=$copied normalized_lf=$normalized_lf skipped_missing=$skipped_missing"
