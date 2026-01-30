set -euo pipefail

DIR="tmp"
PATTERN="*.png"
DRYRUN=0

usage() {
  echo "Usage: $0 [-d DIR] [-p PATTERN] [--dry-run]"
  echo "  -d DIR       Directory to clean (default: tmp)"
  echo "  -p PATTERN   Filename pattern (default: *.png)"
  echo "  --dry-run    Show what would be deleted, do not delete"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) DIR="$2"; shift 2 ;;
    -p) PATTERN="$2"; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "$DIR" ]]; then
  echo "Directory not found: $DIR"
  exit 0
fi

if [[ $DRYRUN -eq 1 ]]; then
  echo "[DRY-RUN] Would delete files matching '$PATTERN' under '$DIR':"
  # List with null safety
  find "$DIR" -type f -name "$PATTERN" -print0 | xargs -0 -r ls -lh || true
  exit 0
fi

# Delete using find's -delete (no arglist expansion)
# If -delete isn't available, fallback to xargs rm
if find "$DIR" -type f -name "$PATTERN" -print -quit | grep -q .; then
  # Prefer -delete (atomic and fast)
  if find "$DIR" -type f -name "$PATTERN" -delete 2>/dev/null; then
    echo "Deleted all files matching '$PATTERN' under '$DIR'."
  else
    # Fallback: xargs (null-delimited)
    echo "Fallback to xargs rm..."
    find "$DIR" -type f -name "$PATTERN" -print0 | xargs -0 -r rm -f
    echo "Deleted all files matching '$PATTERN' under '$DIR'."
  fi
else
  echo "No files matching '$PATTERN' under '$DIR'."
fi
