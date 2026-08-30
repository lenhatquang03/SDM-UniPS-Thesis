#!/usr/bin/env bash
# extract_and_cleanup.sh
# Extract zip archives one at a time, deleting each only after its extraction is
# verified. Sequential by design: peak disk is ONE archive plus its output, not
# the sum of all of them.
#
# USAGE
#   ./extract_and_cleanup.sh <dest_dir> <archive.zip> [archive.zip ...]
#
# TOGGLES
#   SKIP_TEST=1  skip the CRC pre-check (saves one full read per archive)
#   KEEP_ZIP=1   extract but do not delete - use for a dry run
#   VERBOSE=1    list every extracted file
set -uo pipefail

SKIP_TEST="${SKIP_TEST:-0}"
KEEP_ZIP="${KEEP_ZIP:-0}"
VERBOSE="${VERBOSE:-0}"
TIMING_LOG="./unzip_timings.tsv"
# Extra headroom demanded on top of the uncompressed size.
CUSHION=$(( 1024 * 1024 * 1024 ))

# INPUTS
DEST="${1:-}"
[[ -n "$DEST" ]] || { echo "Usage: $0 <dest_dir> <archive.zip> [...]"; exit 1; }
shift
ARCHIVES=("$@")
(( ${#ARCHIVES[@]} > 0 )) || { echo "No archives given."; exit 1; }

mkdir -p "$DEST" || exit 1
[[ -f "$TIMING_LOG" ]] || printf 'timestamp\tarchive\tzip_bytes\tout_bytes\tentries\telapsed_s\n' > "$TIMING_LOG"

declare -a UNZIP_FLAGS=(-o -q)
(( VERBOSE == 1 )) && UNZIP_FLAGS=(-o)

# HELPER FUNCS
separator() { echo; printf '=%.0s' {1..60}; echo; }
human() { numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1} bytes"; }

# unzip -Zt prints e.g. "42 files, 139586437120 bytes uncompressed, ..."
# Field 1 is the entry count, field 3 the uncompressed total. Both come from the
# central directory, so this is exact rather than an estimate.
zip_totals() {              # $1 = archive -> "<entries> <uncompressed_bytes>"
  unzip -Zt "$1" 2>/dev/null | awk '{print $1, $3}'
}

# unzip -q is silent for hours on a large archive. Poll the destination instead
# and report percentage against the known uncompressed total.
watch_progress() {          # $1 = pid to watch, $2 = expected bytes, $3 = label
  local pid="$1" expect="$2" label="$3" before now pct
  before=$(du -sb "$DEST" 2>/dev/null | cut -f1)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    kill -0 "$pid" 2>/dev/null || break
    now=$(du -sb "$DEST" 2>/dev/null | cut -f1)
    if (( expect > 0 )); then
      pct=$(( (now - before) * 100 / expect ))
      echo "     ${label}: $(human $(( now - before ))) / $(human "$expect")  (~${pct}%)"
    else
      echo "     ${label}: $(human $(( now - before ))) written"
    fi
  done
}

echo "${#ARCHIVES[@]} archive(s) -> ${DEST}"
(( SKIP_TEST == 1 )) && echo "SKIP_TEST=1: CRC pre-check disabled."
(( KEEP_ZIP == 1 )) && echo "KEEP_ZIP=1: archives will NOT be deleted."

# MAIN
done_ok=0; failed=0; skipped=0
reclaimed=0
declare -a FAILED_LIST=()
n=${#ARCHIVES[@]}
idx=0

for zip in "${ARCHIVES[@]}"; do
  idx=$(( idx + 1 ))
  separator
  base=$(basename "$zip")
  echo "[${idx}/${n}] ${base}"

  if [[ ! -f "$zip" ]]; then
    echo "  SKIP: ${zip} does not exist."
    skipped=$(( skipped + 1 )); FAILED_LIST+=("${base} (missing)")
    continue
  fi

  zip_bytes=$(stat -c%s "$zip")
  read -r entries out_bytes < <(zip_totals "$zip")
  entries="${entries:-0}"; out_bytes="${out_bytes:-0}"
  if (( entries == 0 )); then
    echo "  SKIP: cannot read central directory - not a zip, or truncated."
    failed=$(( failed + 1 )); FAILED_LIST+=("${base} (unreadable)")
    continue
  fi
  # -Zt counts directory entries too; the verification below counts only files,
  # so report the file count here to keep the two numbers comparable.
  files=$(unzip -Z1 "$zip" | grep -cv '/$')
  echo "  $(human "$zip_bytes") zipped, ${files} files, $(human "$out_bytes") uncompressed"

  # SPACE: refuse rather than fill the disk halfway through extraction.
  avail=$(df -B1 --output=avail "$DEST" | tail -1)
  if (( avail < out_bytes + CUSHION )); then
    echo "  SKIP: need $(human $(( out_bytes + CUSHION ))), have $(human "$avail")."
    skipped=$(( skipped + 1 )); FAILED_LIST+=("${base} (no space)")
    continue
  fi

  # CRC PRE-CHECK: catches a truncated download BEFORE anything is deleted.
  if (( SKIP_TEST == 0 )); then
    echo "  testing archive integrity..."
    if ! unzip -t -q "$zip"; then
      echo "  CRC TEST FAILED - archive is corrupt, leaving it in place."
      failed=$(( failed + 1 )); FAILED_LIST+=("${base} (crc)")
      continue
    fi
    echo "  integrity OK"
  fi

  # EXTRACT with a background progress poller.
  echo "  extracting to ${DEST}..."
  t0=$SECONDS
  unzip "${UNZIP_FLAGS[@]}" "$zip" -d "$DEST" &
  UNZIP_PID=$!
  watch_progress "$UNZIP_PID" "$out_bytes" "$base" &
  WATCH_PID=$!
  wait "$UNZIP_PID"
  rc=$?
  kill "$WATCH_PID" 2>/dev/null
  wait "$WATCH_PID" 2>/dev/null
  elapsed=$(( SECONDS - t0 ))

  if (( rc != 0 )); then
    echo "  unzip exited ${rc} - keeping ${base}."
    failed=$(( failed + 1 )); FAILED_LIST+=("${base} (unzip rc=${rc})")
    continue
  fi

  # VERIFY: entry count from the archive vs files now on disk. Counting rather
  # than comparing bytes, since block rounding makes size comparison noisy.
  listed=$(unzip -Z1 "$zip" | grep -cv '/$')
  present=0
  while IFS= read -r f; do
    [[ -e "${DEST}/${f}" ]] && present=$(( present + 1 ))
  done < <(unzip -Z1 "$zip" | grep -v '/$')
  echo "  verified ${present}/${listed} entries present"

  if (( present != listed )); then
    echo "  INCOMPLETE - keeping ${base}."
    failed=$(( failed + 1 )); FAILED_LIST+=("${base} (${present}/${listed})")
    continue
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "$base" "$zip_bytes" "$out_bytes" "$entries" "$elapsed" >> "$TIMING_LOG"

  # DELETE: only now, with extraction confirmed.
  if (( KEEP_ZIP == 1 )); then
    echo "  KEEP_ZIP=1 - ${base} left in place."
  else
    rm -f "$zip"
    reclaimed=$(( reclaimed + zip_bytes ))
    echo "  deleted ${base} ($(human "$zip_bytes") reclaimed)"
  fi

  echo "  OK in ${elapsed}s"
  done_ok=$(( done_ok + 1 ))
done

# SUMMARY
separator
echo "ARCHIVES"
echo "  Extracted : ${done_ok}"
echo "  Failed    : ${failed}"
echo "  Skipped   : ${skipped}"
if (( ${#FAILED_LIST[@]} > 0 )); then
  printf '    - %s\n' "${FAILED_LIST[@]}"
fi
echo
echo "  Reclaimed : $(human "$reclaimed")"
echo "  Free now  : $(human "$(df -B1 --output=avail "$DEST" | tail -1)") on ${DEST}"
echo "  Timings appended to ${TIMING_LOG}"