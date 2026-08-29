#!/usr/bin/env bash
# Finds the .partNN pieces of one or more ORIGINAL archives across the configured repos, 
# verifies each against the Hub's SHA-256, extracts, and deletes everything it downloaded.
#
# USAGE
#   ./download_and_extract_HG_data.sh <download_dir> [basename ...]
#   ./download_and_extract_HG_data.sh                     # prompts for both
# With no basenames, every archive found across the repos is processed.
# <download_dir> is where parts land, where archives are extracted, and where the
# downloaded zips are deleted from.
#
# STREAM=1 (default): parts are piped straight into bsdtar and deleted as they are
#   consumed, so peak extra disk is ONE part plus the extracted output. Requires
#   bsdtar, which reads zip from a stream; unzip cannot (it seeks to the central
#   directory at the end of the file).
# STREAM=0: download every part, concatenate, verify the whole-file hash from the
#   <base>.sha256 sidecar, unzip, then delete the merged zip. Needs ~3x the space
#   but keeps every part on disk until extraction succeeds.
#
# PROGRESS: every read that can take minutes goes through `dd status=progress`
# rather than cat/sha256sum, so no phase is silent. dd writes to stderr using
# carriage returns - `tail -f` renders it correctly, `less` does not.
set -uo pipefail

# Searched in order; the first repo containing a given path wins.
REPOS=("culacgiontan0312/UniPS" "HUST-CVLab-PS/UniPS")
# REPOS=("culacgiontan0312/UniPS") # Trial run

RETRIES=3
RETRY_WAIT=30
STREAM="${STREAM:-1}"
# VERBOSE_EXTRACT=1 lists every extracted file. Off by default: a 130GiB archive
# is thousands of names and it buries everything else in the log. In STREAM=1 the
# dd progress feeding bsdtar already shows that extraction is moving.
VERBOSE_EXTRACT="${VERBOSE_EXTRACT:-0}"
# Set AUTO_APT=1 to install bsdtar without prompting (useful under nohup).
AUTO_APT="${AUTO_APT:-}"
TIMING_LOG="./hf_download_timings.tsv"

# USER INPUTS
# First positional arg is the download dir; the rest are basenames.
DOWNLOAD_DIR="${1:-}"
if [[ -n "$DOWNLOAD_DIR" ]]; then
  # Remove the 1st positional args
  shift
  # Bash array of all remaining positional args
  BASES=("$@")
else
  read -r -p "Download dir (absolute path; parts, extraction and cleanup all happen here): " DOWNLOAD_DIR
  read -r -p "Archive basenames (blank for every archive in the repos): " -a BASES
fi

[[ -n "$DOWNLOAD_DIR" ]] || { echo "No download dir given."; exit 1; }
# No basenames means every archive in the repos.
WANT_ALL=$(( ${#BASES[@]} == 0 ))

# Parts are staged in a hidden subdir so they never collide with extracted output.
WORK="${DOWNLOAD_DIR}/.parts"
mkdir -p "$DOWNLOAD_DIR" "$WORK" || exit 1
[[ -f "$TIMING_LOG" ]] || printf 'timestamp\tpart\tbytes\tdownload_ms\n' > "$TIMING_LOG"

# Extraction verbosity, resolved once into argument arrays.
declare -a BSDTAR_FLAGS=(-xf)
declare -a UNZIP_FLAGS=(-q -o)
if (( VERBOSE_EXTRACT == 1 )); then
  BSDTAR_FLAGS=(-xvf)
  UNZIP_FLAGS=(-o)
fi

# HELPER FUNCS
separator() { echo; printf '=%.0s' {1..60}; echo; }
human() { numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1} bytes"; }
now_ms() { date +%s%3N; }

hash_file() {               # $1 = path -> sha256 on stdout, progress on stderr
  # dd instead of a bare sha256sum so a 40GiB verify is not a silent 20 minutes.
  dd if="$1" bs=4M status=progress | sha256sum | cut -d' ' -f1
}

# bsdtar is the only zip reader that works on a stream. Without it, fall back.
ensure_bsdtar() {
  command -v bsdtar >/dev/null 2>&1 && return 0
  (( STREAM == 0 )) && return 1
  echo "bsdtar not found. Streaming extraction needs it (package: libarchive-tools)."
  local ans="$AUTO_APT"
  # Interactive prompt to whether install bdstar with apt
  if [[ -z "$ans" ]]; then
    if [[ -r /dev/tty ]]; then
      read -r -p "Install it with apt now? [y/N] " ans < /dev/tty
    else
      ans="n"
    fi
  fi
  # 'esac' is 'case' reversed, ';;' marks the end of a case branch
  case "$ans" in
    y|Y|1|yes)
      if [[ $EUID -eq 0 ]]; then
        apt-get update -qq && apt-get install -y libarchive-tools
      else
        sudo apt-get update -qq && sudo apt-get install -y libarchive-tools
      fi
      ;;
  esac
  # Checking again if it is truly installed
  command -v bsdtar >/dev/null 2>&1
}

if (( STREAM == 1 )) && ! ensure_bsdtar; then
  echo "Falling back to STREAM=0 (download all parts, merge, unzip)."
  echo "This needs roughly 3x the space of the largest archive."
  STREAM=0
fi

download_one() {            # $1 = repo, $2 = path in repo -> file lands at $WORK/$2
  local repo="$1" path="$2" attempt=1
  while (( attempt <= RETRIES )); do
    if HF_XET_HIGH_PERFORMANCE=1 hf download "$repo" "$path" \
         --repo-type dataset --local-dir "$WORK" >&2; then
      return 0
    fi
    echo "  Download failed for ${path} (attempt ${attempt}/${RETRIES})" >&2
    (( attempt < RETRIES )) && sleep "$RETRY_WAIT"
    attempt=$(( attempt + 1 ))
  done
  return 1
}

# REMOTE STATE
# Use path -> Store repo / filename / size / sha256, across every configured repo. 
# First repo wins on a duplicate path, matching the REPOS search order.
declare -A P_REPO=() P_SIZE=() P_SHA=()
while IFS=$'\t' read -r rrepo rpath rsize rsha; do
  if [[ -n "$rpath" && -z "${P_REPO[$rpath]:-}" ]]; then
    P_REPO["$rpath"]="$rrepo"
    P_SIZE["$rpath"]="$rsize"
    [[ -n "$rsha" ]] && P_SHA["$rpath"]="$rsha"
  fi
done < <(python3 - "${REPOS[@]}" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
for repo in sys.argv[1:]:
    try:
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
    except Exception as e:
        print(f"could not read {repo}: {e}", file=sys.stderr)
        continue
    for s in info.siblings:
        if s.size is None:
            continue
        sha = s.lfs.sha256 if s.lfs and s.lfs.sha256 else ""
        print(f"{repo}\t{s.rfilename}\t{s.size}\t{sha}")
PY
)
echo "${#P_SIZE[@]} file(s) visible across ${#REPOS[@]} repo(s)."

# No basenames given: derive the set of original archive names from the Hub.
# A .partNN suffix is stripped; .sha256 sidecars and dotfiles are not archives.
if (( WANT_ALL == 1 )); then
  declare -A SEEN=()
  # !P_SIZE[@] gives the keys/indices of the associative/indexed P_SIZE array.
  for k in "${!P_SIZE[@]}"; do
    # Extract the basename
    name="${k##*/}"
    case "$name" in
      *.sha256|.*|README*|*.md) continue ;;
      *.part[0-9][0-9]) SEEN["${name%.part[0-9][0-9]}"]=1 ;;
      *)                SEEN["$name"]=1 ;;
    esac
  done
  # Put the basenames into the indexed array BASES
  mapfile -t BASES < <(printf '%s\n' "${!SEEN[@]}" | sort)
  echo "No basenames given -> ${#BASES[@]} archive(s): ${BASES[*]}"
fi

(( ${#BASES[@]} > 0 )) || { echo "Nothing to download."; exit 1; }
echo "Download dir: ${DOWNLOAD_DIR}   (parts staged in ${WORK})"
(( VERBOSE_EXTRACT == 1 )) && echo "VERBOSE_EXTRACT=1: every extracted filename will be listed."

# MAIN
extracted=0; efailed=0; missing=0
parts_ok=0; parts_sizeonly=0
dl_bytes=0; dl_ms=0
declare -a FAILED_LIST=()
n_bases=${#BASES[@]}
b_idx=0

for base in "${BASES[@]}"; do
  b_idx=$(( b_idx + 1 ))
  separator
  echo "[${b_idx}/${n_bases}] ${base}"

  # DISCOVERY: collect <anything>/<base>.partNN, ordered by NN.
  # Glob matching rather than regex so dots in the basename need no escaping.
  mapfile -t PARTS < <(
    for k in "${!P_SIZE[@]}"; do
      name="${k##*/}"
      case "$name" in
        "${base}".part[0-9][0-9]) printf '%s\t%s\n' "${name##*.part}" "$k" ;;
      esac
    done | sort -n | cut -f2
  )

  # A file small enough to skip chunking was uploaded whole.
  if (( ${#PARTS[@]} == 0 )); then
    for k in "${!P_SIZE[@]}"; do
      [[ "${k##*/}" == "$base" ]] && PARTS=("$k") && break
    done
  fi

  if (( ${#PARTS[@]} == 0 )); then
    echo "  NOT FOUND in any configured repo."
    missing=$(( missing + 1 )); FAILED_LIST+=("${base} (not found)")
    continue
  fi

  # GAP CHECK: cat would silently splice 0,1,2,4 into a corrupt archive.
  gap=0
  if [[ "${PARTS[0]}" == *.part* ]]; then
    for (( i = 0; i < ${#PARTS[@]}; i++ )); do
      want=$(printf '%s.part%02d' "$base" "$i")
      [[ "${PARTS[i]##*/}" == "$want" ]] || { echo "  GAP: expected ${want}"; gap=1; break; }
    done
  fi
  if (( gap == 1 )); then
    echo "  Refusing to extract an incomplete set."
    efailed=$(( efailed + 1 )); FAILED_LIST+=("${base} (missing parts)")
    continue
  fi

  # Sizes: every part but the last should be identical. If the LAST one matches
  # that size too, the set may be truncated - suggestive, not conclusive.
  total=0
  for p in "${PARTS[@]}"; do total=$(( total + ${P_SIZE[$p]} )); done
  if (( ${#PARTS[@]} > 1 )); then
    first_sz=${P_SIZE[${PARTS[0]}]}
    last_sz=${P_SIZE[${PARTS[-1]}]}
    (( last_sz == first_sz )) && echo "  NOTE: last part is full-size; a further part may be missing."
  fi
  echo "  ${#PARTS[@]} part(s), $(human "$total") total, from ${P_REPO[${PARTS[0]}]}"

  # Whole-file sidecar, if the upload side produced one.
  dirpart="${PARTS[0]%/*}"
  [[ "$dirpart" == "${PARTS[0]}" ]] && sidecar="${base}.sha256" || sidecar="${dirpart}/${base}.sha256"
  whole_expect=""
  if [[ -n "${P_SIZE[$sidecar]:-}" ]]; then
    if download_one "${P_REPO[$sidecar]}" "$sidecar"; then
      whole_expect=$(cut -d' ' -f1 < "${WORK}/${sidecar}")
      rm -f "${WORK}/${sidecar}"
      echo "  sidecar hash: ${whole_expect:0:16}..."
    fi
  else
    echo "  NOTE: no ${base}.sha256 sidecar; per-part hashes only."
  fi

  # SPACE: streaming needs one part plus the extracted output; merging needs
  # every part plus a merged copy plus the output. 1.1x is a guess for the
  # uncompressed size - zip's real figure needs the central directory.
  avail=$(df -B1 --output=avail "$DOWNLOAD_DIR" | tail -1)
  largest=0
  for p in "${PARTS[@]}"; do (( P_SIZE[$p] > largest )) && largest=${P_SIZE[$p]}; done
  est_out=$(( total * 11 / 10 ))
  if (( STREAM == 1 )); then need=$(( largest + est_out )); else need=$(( total * 2 + est_out )); fi
  echo "  need ~$(human "$need"), have $(human "$avail")"
  if (( avail < need )); then
    echo "  WARNING: estimated space is short. Continuing anyway - the estimate is rough."
  fi

  ok=1
  if (( STREAM == 1 )); then
    # Feed parts into bsdtar in order, hashing the concatenation on the way past.
    # A FIFO (rather than a process substitution) lets us wait on the hasher.
    SHA_FIFO=$(mktemp -u); mkfifo "$SHA_FIFO"
    WHOLE_OUT=$(mktemp)
    STATUS=$(mktemp)
    sha256sum < "$SHA_FIFO" | cut -d' ' -f1 > "$WHOLE_OUT" &
    # Extract the latest background process' PID (cut)
    SHA_PID=$!

    (
      pn=0
      for p in "${PARTS[@]}"; do
        pn=$(( pn + 1 ))
        echo "  -> [${pn}/${#PARTS[@]}] ${p##*/} ($(human "${P_SIZE[$p]}"))" >&2
        t0=$(now_ms)
        download_one "${P_REPO[$p]}" "$p" || { echo "DLFAIL $p" >> "$STATUS"; exit 1; }
        t1=$(now_ms)
        lp="${WORK}/${p}"
        got=$(stat -c%s "$lp" 2>/dev/null || echo 0)
        if (( got != ${P_SIZE[$p]} )); then
          echo "SIZEFAIL $p" >> "$STATUS"; exit 1
        fi
        exp="${P_SHA[$p]:-}"
        if [[ -n "$exp" ]]; then
          echo "     verifying SHA-256..." >&2
          act=$(hash_file "$lp")
          [[ "$act" == "$exp" ]] || { echo "HASHFAIL $p" >> "$STATUS"; exit 1; }
          echo "HASHOK $p" >> "$STATUS"
        else
          echo "SIZEONLY $p" >> "$STATUS"
        fi
        printf 'DL\t%s\t%s\n' "${P_SIZE[$p]}" "$(( t1 - t0 ))" >> "$STATUS"
        printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$p" "${P_SIZE[$p]}" "$(( t1 - t0 ))" >> "$TIMING_LOG"
        # dd rather than cat: this is the extraction feed, and its progress line is
        # the only signal that bsdtar is making headway.
        echo "     feeding to bsdtar..." >&2
        dd if="$lp" bs=4M status=progress
        # Each part is deleted the moment it has been consumed by the named pipe $SHA_FIFO.
        rm -f "$lp"
      done
    ) | tee "$SHA_FIFO" | bsdtar "${BSDTAR_FLAGS[@]}" - -C "$DOWNLOAD_DIR"
    # Record the exit status of the most recent command. 
    # With 'set -o pipefail', 'rc' is 0 if the pipe succeeds, else the exit status of the right-most failing command.
    rc=$?
    # Bash knows 'cut' is a part of the pipeline, wait on its PID = wait for the pipeline completion
    wait "$SHA_PID"
    whole_actual=$(cat "$WHOLE_OUT" 2>/dev/null)
    rm -f "$SHA_FIFO" "$WHOLE_OUT"

    # Tally what the subshell recorded; its variables did not survive the pipe.
    while read -r tag a b; do
      case "$tag" in
        HASHOK)   parts_ok=$(( parts_ok + 1 )) ;;
        SIZEONLY) parts_sizeonly=$(( parts_sizeonly + 1 )) ;;
        DL)       dl_bytes=$(( dl_bytes + a )); dl_ms=$(( dl_ms + b )) ;;
        DLFAIL|SIZEFAIL|HASHFAIL) echo "  ${tag} on ${a}"; ok=0 ;;
      esac
    done < "$STATUS"
    rm -f "$STATUS"

    (( rc == 0 )) || { echo "  bsdtar exited ${rc}"; ok=0; }
    if [[ -n "$whole_expect" && -n "$whole_actual" ]]; then
      if [[ "$whole_expect" == "$whole_actual" ]]; then
        echo "  whole-file SHA-256 verified."
      else
        echo "  WHOLE-FILE HASH MISMATCH"; ok=0
      fi
    fi
  else
    # STREAM=0: fetch everything, merge, verify, unzip, then reclaim the space.
    merged="${WORK}/${base}"
    : > "$merged"
    pn=0
    for p in "${PARTS[@]}"; do
      pn=$(( pn + 1 ))
      echo "  -> [${pn}/${#PARTS[@]}] ${p##*/} ($(human "${P_SIZE[$p]}"))"
      t0=$(now_ms)
      download_one "${P_REPO[$p]}" "$p" || { ok=0; break; }
      t1=$(now_ms)
      lp="${WORK}/${p}"
      got=$(stat -c%s "$lp" 2>/dev/null || echo 0)
      (( got == ${P_SIZE[$p]} )) || { echo "  SIZE MISMATCH on ${p}"; ok=0; break; }
      exp="${P_SHA[$p]:-}"
      if [[ -n "$exp" ]]; then
        echo "     verifying SHA-256..."
        act=$(hash_file "$lp")
        [[ "$act" == "$exp" ]] || { echo "  HASH MISMATCH on ${p}"; ok=0; break; }
        parts_ok=$(( parts_ok + 1 ))
      else
        parts_sizeonly=$(( parts_sizeonly + 1 ))
      fi
      dl_bytes=$(( dl_bytes + got )); dl_ms=$(( dl_ms + t1 - t0 ))
      printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$p" "$got" "$(( t1 - t0 ))" >> "$TIMING_LOG"
      # dd rather than cat so the append shows progress; >> keeps the merge going.
      echo "     appending to ${base}..."
      dd if="$lp" bs=4M status=progress >> "$merged"
      # Part folded into the merged archive; its copy is no longer needed.
      rm -f "$lp"
    done

    if (( ok == 1 )) && [[ -n "$whole_expect" ]]; then
      echo "  verifying merged archive..."
      whole_actual=$(hash_file "$merged")
      if [[ "$whole_expect" == "$whole_actual" ]]; then
        echo "  whole-file SHA-256 verified."
      else
        echo "  WHOLE-FILE HASH MISMATCH"; ok=0
      fi
    fi

    if (( ok == 1 )); then
      echo "  extracting..."
      if unzip "${UNZIP_FLAGS[@]}" "$merged" -d "$DOWNLOAD_DIR"; then
        # Delete the merged zip as soon as extraction succeeds, so peak usage
        # drops back to just the extracted tree.
        rm -f "$merged"
        echo "  removed ${base} ($(human "$total") reclaimed)"
      else
        echo "  unzip failed"; ok=0
      fi
    fi
    # On failure the merged zip is kept deliberately: re-running unzip is cheap,
    # re-downloading it is not. Delete it by hand once you are done with it.
    if (( ok == 0 )) && [[ -s "$merged" ]]; then
      echo "  kept ${merged} for retry - delete it manually when finished."
    fi
  fi

  if (( ok == 1 )); then
    echo "  OK: ${base} extracted to ${DOWNLOAD_DIR}"
    extracted=$(( extracted + 1 ))
  else
    echo "  FAILED: ${base}"
    efailed=$(( efailed + 1 )); FAILED_LIST+=("$base")
  fi
done

# Remove the staging dir only if it is empty; anything left is a deliberate keep.
find "$WORK" -type d -empty -delete 2>/dev/null
if [[ -d "$WORK" ]]; then
  echo
  echo "NOTE: ${WORK} is not empty - leftover downloads from a failed archive."
fi

# SUMMARY
separator
echo "ARCHIVES"
echo "  Extracted : ${extracted}"
echo "  Failed    : ${efailed}"
echo "  Not found : ${missing}"
if (( ${#FAILED_LIST[@]} > 0 )); then
  printf '    - %s\n' "${FAILED_LIST[@]}"
fi

echo
echo "PARTS"
echo "  SHA-256 verified : ${parts_ok}"
echo "  Size only        : ${parts_sizeonly} (no hash published for these)"

echo
echo "THROUGHPUT"
if (( dl_bytes == 0 || dl_ms == 0 )); then
  echo "  Nothing downloaded."
else
  awk -v b="$dl_bytes" -v ms="$dl_ms" 'BEGIN {
    rate = b / (ms / 1000)
    printf "  Download : %.1f Mbps  (%.2f GiB in %.0f s)\n",
           rate * 8 / 1e6, b / 1073741824, ms / 1000
  }'
fi
echo "  Per-part timings appended to ${TIMING_LOG}"
echo "  Mode: STREAM=${STREAM}   Dir: ${DOWNLOAD_DIR}"