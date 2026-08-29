#!/usr/bin/env bash
# chunk_and_upload_HG_data.sh
# Carve oversized archives into <50GiB parts and upload each to a HF dataset repo,
# one part at a time so peak disk usage stays at one chunk.
#
# LAYOUT: parts are stored under a subdirectory chosen by filename prefix
# (see subdir_for), alongside a <base>.sha256 sidecar holding the whole-file hash
# of the ORIGINAL archive. download_and_extract_HG_data.sh consumes both.
#
# RESUME: safe to re-run. Parts already on the Hub are skipped after their SHA-256
# is matched against the Hub's, falling back to a size-only check when no local
# hash is known (see VERIFY_HASH).
# WARNING: always re-run with the SAME CHUNK_GB. A different chunk size re-splits
# everything; parts from the previous size are NOT cleaned up and would be silently
# concatenated by `cat *.part*` on the download side. Wipe the repo first instead.
set -uo pipefail

RETRIES=3
RETRY_WAIT=30

# ETA MEASUREMENT
# Size the closing projection is computed for, in GiB.
TARGET_GIB=183
# Per-part timings, appended across runs. VN international routing varies by hour,
# so several samples at different times beat one.
TIMING_LOG="./hf_upload_timings.tsv"

# HASH VERIFICATION
# Local hashes, keyed on source path + size + mtime + part name, so a regenerated
# source file invalidates its own entries instead of matching stale hashes.
HASH_CACHE="./hf_upload_hashes.tsv"
# VERIFY_HASH=1 re-reads the source byte range to verify a resumed part when its
# hash is not in the cache (fresh clone, different machine). Costs one full read
# of every resumed file; off by default.
VERIFY_HASH="${VERIFY_HASH:-0}"
# WHOLE_HASH=1 computes the whole-file SHA-256 of each source archive and uploads
# it as <base>.sha256, so the download side can verify the reassembly end to end.
# Costs one extra full read per file, once (cached afterwards). WHOLE_HASH=0 skips
# it; the download side then falls back to per-part hashes only.
WHOLE_HASH="${WHOLE_HASH:-1}"

# USER INPUTS
# Check if HF_REPO is already set, else read from input
REPO="${HF_REPO:-}"
[[ -n "$REPO" ]] || read -r -p "Target HF dataset repo (e.g. quangln21/unips-extra): " REPO
# Specify files to chunk and chunk size
read -r -p "Specify files to chunk (ABSOLUTE PATHS, space-separated): " -a FILES
read -r -p "Specify chunk size (GiB): " CHUNK_GB

# INPUT VALIDATION
[[ "$CHUNK_GB" =~ ^[0-9]+$ ]] && (( CHUNK_GB > 0 )) \
  || { echo "Chunk size must be a positive integer."; exit 1; }
# HG datafile size ceilling
(( CHUNK_GB <= 45 )) \
  || { echo "HF rejects LFS files above 50GiB. Use 45 or less."; exit 1; }
# Number of input files
(( ${#FILES[@]} > 0 )) || { echo "No files given."; exit 1; }

CHUNK_BYTES=$(( CHUNK_GB * 1024 * 1024 * 1024 ))
MANIFEST=$(mktemp)
[[ -f "$TIMING_LOG" ]] || printf 'timestamp\tpart\tbytes\tcarve_ms\tupload_ms\n' > "$TIMING_LOG"

echo "${#FILES[@]} path(s) received. Chunk size ${CHUNK_GB} GiB. Target: ${REPO}"
(( VERIFY_HASH == 1 )) && echo "VERIFY_HASH=1: uncached resumed parts will be re-hashed from source."
(( WHOLE_HASH == 0 )) && echo "WHOLE_HASH=0: no .sha256 sidecars will be produced."

# HELPER FUNCS
separator() { echo; printf '=%.0s' {1..60}; echo; }
human() { numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1} bytes"; }
# e.g. human($((10 * 1024 * 1024))) -> 10MiB
# %s = seconds since the Epoch (1970-01-01 00:00 UTC) + %3N = the first 3 digits of $N (nanoseconds) = Milliseconds
now_ms() { date +%s%3N; }

subdir_for() {              # $1 = absolute source path -> repo subdirectory
  # Group by filename prefix so the repo mirrors the dataset's structure rather
  # than dumping every part at the root.
  local b; b=$(basename "$1")
  case "$b" in
    obj_*)    printf 'polarps' ;;
    hdlong_*) printf 'hdlong' ;;
    merlmix*) printf 'merlmix' ;;
    # Fallback: the source file's own parent directory name.
    *)        dirname "$1" | awk -F/ '{printf "%s", $NF}' ;;
  esac
}

upload_one() {              # $1 = local path, $2 = name in repo
  local src="$1" name="$2" attempt=1
  while (( attempt <= RETRIES )); do
    if HF_XET_HIGH_PERFORMANCE=1 hf upload "$REPO" "$src" "$name" --repo-type dataset; then
      return 0
    fi
    # Retry HG upload 3 times, sleep 30s between each
    echo "  Upload failed for ${name} (attempt ${attempt}/${RETRIES})"
    (( attempt < RETRIES )) && sleep "$RETRY_WAIT"
    attempt=$(( attempt + 1 ))
  done
  return 1
}

carve() {                   # $1 = src, $2 = dest, $3 = offset, $4 = length, $5 = hash out-file
  # Assume that skip=10 count=5
  # With the iflag arg, 'dd' skips the first 10 bytes of if and reads the next 5 bytes.
  # Without the iflag arg, 'dd' skips the first 10*4M bytes of if and reads the next 5*4M bytes.
  # dd writes to stdout instead of of=; tee lands the chunk on disk (truncating, as
  # of= did) while sha256sum hashes the same bytes already in memory. sha256sum runs
  # far faster than the source read, so this costs no extra wall clock.
  # pipefail (set at the top) makes the pipeline fail if any stage fails.
  dd if="$1" bs=4M iflag=skip_bytes,count_bytes skip="$3" count="$4" status=progress \
    | tee "$2" | sha256sum | cut -d' ' -f1 > "$5"
}

hash_range() {              # $1 = src, $2 = offset, $3 = length -> sha256 on stdout
  # No tee: verifies a byte range without writing anything to disk.
  # status=progress goes to stderr, so the command substitution around this call
  # still captures nothing but the hash. Used for VERIFY_HASH re-reads and for the
  # whole-file sidecar hash - both slow enough to need a visible progress line.
  dd if="$1" bs=4M iflag=skip_bytes,count_bytes skip="$2" count="$3" status=progress \
    | sha256sum | cut -d' ' -f1
}

remote_sha() {              # $1 = path in repo -> sha256 on stdout (empty if unknown)
  python3 - "$REPO" "$1" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    info = HfApi().repo_info(sys.argv[1], repo_type="dataset", files_metadata=True)
    for s in info.siblings:
        if s.rfilename == sys.argv[2]:
            print(s.lfs.sha256 if s.lfs and s.lfs.sha256 else "")
            break
except Exception:
    pass
PY
}

cache_key() {               # $1 = src path, $2 = src size, $3 = src mtime, $4 = part name
  printf '%s|%s|%s|%s' "$1" "$2" "$3" "$4"
}

# HASH CACHE (local, persistent)
declare -A HASHCACHE=()
if [[ -f "$HASH_CACHE" ]]; then
  while IFS=$'\t' read -r hk hv; do
    [[ -n "$hk" ]] && HASHCACHE["$hk"]="$hv"
  done < "$HASH_CACHE"
fi
echo "${#HASHCACHE[@]} cached hash(es) in ${HASH_CACHE}."

# REMOTE STATE (for resume)
# Snapshot what is already on the Hub: path -> size, and path -> sha256.
# On failure, proceed with empty maps: costs redundant work, risks nothing.
declare -A REMOTE=()
declare -A REMOTE_SHA=()
while IFS=$'\t' read -r rname rsize rsha; do
  if [[ -n "$rname" ]]; then
    REMOTE["$rname"]="$rsize"
    [[ -n "$rsha" ]] && REMOTE_SHA["$rname"]="$rsha"
  fi
done < <(python3 - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
try:
    info = HfApi().repo_info(sys.argv[1], repo_type="dataset", files_metadata=True)
except Exception as e:
    print(f"could not read remote state: {e}", file=sys.stderr)
    sys.exit(0)
for s in info.siblings:
    if s.size is not None:
        sha = s.lfs.sha256 if s.lfs and s.lfs.sha256 else ""
        print(f"{s.rfilename}\t{s.size}\t{sha}")
PY
)
echo "${#REMOTE[@]} file(s) already present in ${REPO}."

# MAIN
# CHUNK-LEVEL COUNTERS
uploaded=0; resumed=0; failed=0
# FILE-LEVEL COUNTERS
complete=0; incomplete=0; skipped=0
# VERIFICATION COUNTERS
hash_ok=0; size_only=0; hash_bad=0; sidecars=0
# THROUGHPUT ACCUMULATORS
# Only parts that completed a phase this run contribute. Resumed parts would add
# bytes without adding time (rate -> infinity); failed parts would add time
# without adding bytes.
carve_bytes=0; carve_ms=0
up_bytes=0; up_ms=0
declare -a FAILED_LIST=()
declare -a INCOMPLETE_LIST=()

for file in "${FILES[@]}"; do
  separator
  if [[ ! -f "$file" ]]; then
    echo "SKIP: ${file} does not exist. Skipped."
    skipped=$(( skipped + 1 ))
    continue
  fi

  file_bytes=$(stat -c%s "$file")
  file_mtime=$(stat -c%Y "$file")
  dir=$(dirname "$file")
  base=$(basename "$file")
  subdir=$(subdir_for "$file")
  sidecar="${subdir}/${base}.sha256"
  # Cleared by any part failure or by the disk-full break
  file_ok=1
  whole_sha=""
  wkey=$(cache_key "$file" "$file_bytes" "$file_mtime" "__WHOLE__")
  echo "${base} -> ${REPO}:${subdir}/"

  # Small enough to send whole: chunking would only waste disk and time.
  if (( file_bytes <= CHUNK_BYTES )); then
    repo_path="${subdir}/${base}"
    ckey=$(cache_key "$file" "$file_bytes" "$file_mtime" "$base")
    local_sha="${HASHCACHE[$ckey]:-}"

    if [[ "${REMOTE[$repo_path]:-}" == "$file_bytes" ]]; then
      # RESUME: present at the right size. Confirm by hash where possible.
      if [[ -z "$local_sha" && "$VERIFY_HASH" == "1" ]]; then
        echo "  re-hashing ${base} from source (VERIFY_HASH=1)..."
        local_sha=$(hash_range "$file" 0 "$file_bytes")
      fi
      rsha="${REMOTE_SHA[$repo_path]:-}"
      if [[ -n "$local_sha" && -n "$rsha" ]]; then
        if [[ "$local_sha" == "$rsha" ]]; then
          echo "${base} already on the Hub, hash verified - skipping."
          printf '%s\t%s\t%s\n' "$repo_path" "$file_bytes" "$local_sha" >> "$MANIFEST"
          resumed=$(( resumed + 1 )); complete=$(( complete + 1 )); hash_ok=$(( hash_ok + 1 ))
          # A single-part file's own hash IS the whole-file hash.
          whole_sha="$local_sha"
          if (( WHOLE_HASH == 1 )) && [[ -z "${REMOTE[$sidecar]:-}" ]]; then
            SC_TMP=$(mktemp); printf '%s  %s\n' "$whole_sha" "$base" > "$SC_TMP"
            upload_one "$SC_TMP" "$sidecar" && sidecars=$(( sidecars + 1 ))
            rm -f "$SC_TMP"
          fi
          continue
        fi
        # Same size, different content: re-upload rather than skip.
        echo "  HASH MISMATCH on ${base} - re-uploading."
        hash_bad=$(( hash_bad + 1 ))
      else
        echo "${base} already on the Hub at $(human "$file_bytes") - skipping (size only)."
        printf '%s\t%s\t%s\n' "$repo_path" "$file_bytes" "-" >> "$MANIFEST"
        resumed=$(( resumed + 1 )); complete=$(( complete + 1 )); size_only=$(( size_only + 1 ))
        continue
      fi
    fi

    echo "${base} is $(human "$file_bytes") - uploading whole."
    [[ -n "$local_sha" ]] || local_sha=$(hash_range "$file" 0 "$file_bytes")
    printf '%s\t%s\n' "$ckey" "$local_sha" >> "$HASH_CACHE"
    printf '%s\t%s\t%s\n' "$repo_path" "$file_bytes" "$local_sha" >> "$MANIFEST"
    whole_sha="$local_sha"
    t0=$(now_ms)
    if upload_one "$file" "$repo_path"; then
      t1=$(now_ms)
      # Confirm the Hub stored exactly what we sent.
      rsha=$(remote_sha "$repo_path")
      if [[ -n "$rsha" && "$rsha" != "$local_sha" ]]; then
        echo "  POST-UPLOAD HASH MISMATCH on ${repo_path}"
        failed=$(( failed + 1 )); FAILED_LIST+=("${repo_path} (hash)")
        hash_bad=$(( hash_bad + 1 )); file_ok=0
      else
        uploaded=$(( uploaded + 1 )); complete=$(( complete + 1 ))
        [[ -n "$rsha" ]] && hash_ok=$(( hash_ok + 1 )) || size_only=$(( size_only + 1 ))
        # No carve phase in this branch, so carve_ms is logged as 0.
        up_bytes=$(( up_bytes + file_bytes )); up_ms=$(( up_ms + t1 - t0 ))
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "$(date -Is)" "$repo_path" "$file_bytes" 0 "$(( t1 - t0 ))" >> "$TIMING_LOG"
      fi
    else
      failed=$(( failed + 1 )); FAILED_LIST+=("$repo_path")
      file_ok=0
    fi
    if (( file_ok == 1 && WHOLE_HASH == 1 )) && [[ -z "${REMOTE[$sidecar]:-}" ]]; then
      SC_TMP=$(mktemp); printf '%s  %s\n' "$whole_sha" "$base" > "$SC_TMP"
      upload_one "$SC_TMP" "$sidecar" && sidecars=$(( sidecars + 1 ))
      rm -f "$SC_TMP"
    fi
    (( file_ok == 1 )) || { incomplete=$(( incomplete + 1 )); INCOMPLETE_LIST+=("$base"); }
    continue
  fi

  # WHOLE-FILE HASH for the sidecar. One extra sequential read of the source,
  # done once and cached; skipped entirely if the sidecar is already on the Hub.
  # hash_range over the whole file rather than a bare sha256sum, so the read shows
  # dd's progress line instead of going silent for tens of minutes.
  if (( WHOLE_HASH == 1 )) && [[ -z "${REMOTE[$sidecar]:-}" ]]; then
    whole_sha="${HASHCACHE[$wkey]:-}"
    if [[ -z "$whole_sha" ]]; then
      echo "  computing whole-file SHA-256 of ${base} ($(human "$file_bytes"))..."
      whole_sha=$(hash_range "$file" 0 "$file_bytes")
      printf '%s\t%s\n' "$wkey" "$whole_sha" >> "$HASH_CACHE"
    fi
  fi

  # Ceiling division. file_bytes/CHUNK_BYTES + 1 produces an empty trailing
  # chunk whenever the size is an exact multiple.
  num_chunks=$(( (file_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES ))
  echo "${base} is $(human "$file_bytes") -> ${num_chunks} chunk(s)"

  for (( i = 0; i < num_chunks; i++ )); do
    chunk_name=$(printf '%s.part%02d' "$base" "$i")
    repo_path="${subdir}/${chunk_name}"
    chunk_path="${dir}/${chunk_name}"
    offset=$(( i * CHUNK_BYTES ))
    this_bytes=$(( file_bytes - offset ))
    # Enforce this_bytes <= $CHUNK_BYTES
    (( this_bytes > CHUNK_BYTES )) && this_bytes=$CHUNK_BYTES

    ckey=$(cache_key "$file" "$file_bytes" "$file_mtime" "$chunk_name")
    local_sha="${HASHCACHE[$ckey]:-}"

    # RESUME: already on the Hub at the expected size, so skip the carve entirely.
    if [[ "${REMOTE[$repo_path]:-}" == "$this_bytes" ]]; then
      if [[ -z "$local_sha" && "$VERIFY_HASH" == "1" ]]; then
        echo "  re-hashing ${chunk_name} from source (VERIFY_HASH=1)..."
        local_sha=$(hash_range "$file" "$offset" "$this_bytes")
      fi
      rsha="${REMOTE_SHA[$repo_path]:-}"
      if [[ -n "$local_sha" && -n "$rsha" ]]; then
        if [[ "$local_sha" == "$rsha" ]]; then
          echo "[$(( i + 1 ))/${num_chunks}] ${chunk_name} on the Hub, hash verified - skipping."
          printf '%s\t%s\t%s\n' "$repo_path" "$this_bytes" "$local_sha" >> "$MANIFEST"
          resumed=$(( resumed + 1 )); hash_ok=$(( hash_ok + 1 ))
          continue
        fi
        # Right size, wrong content: fall through and re-carve.
        echo "  HASH MISMATCH on ${chunk_name} - re-uploading."
        hash_bad=$(( hash_bad + 1 ))
      else
        echo "[$(( i + 1 ))/${num_chunks}] ${chunk_name} already on the Hub - skipping (size only)."
        printf '%s\t%s\t%s\n' "$repo_path" "$this_bytes" "-" >> "$MANIFEST"
        resumed=$(( resumed + 1 )); size_only=$(( size_only + 1 ))
        continue
      fi
    fi

    # avail = Available disk space in 1-byte blocks 
    avail=$(df -B1 --output=avail "$dir" | tail -1)
    # If the available space is not enough to accomodate the to-be splitted chunk + 1GiB,
    # abort the split for that specific file
    if (( avail < this_bytes + 1073741824 )); then
      echo "  ABORT ${base}: ${dir} has $(human "$avail") free, need $(human "$this_bytes")."
      printf '%s\t%s\t%s\n' "$repo_path" "$this_bytes" "-" >> "$MANIFEST"
      failed=$(( failed + 1 )); FAILED_LIST+=("${chunk_name} (disk full)")
      file_ok=0
      break
    fi

    echo "[$(( i + 1 ))/${num_chunks}] carving ${chunk_name} ($(human "$this_bytes"))"
    t0=$(now_ms)
    SHA_TMP=$(mktemp)
    carve "$file" "$chunk_path" "$offset" "$this_bytes" "$SHA_TMP"
    carve_rc=$?
    t1=$(now_ms)
    local_sha=$(cat "$SHA_TMP" 2>/dev/null)
    rm -f "$SHA_TMP"

    # If the splitting fails for a chunk, continue with the next chunk
    got=$(stat -c%s "$chunk_path" 2>/dev/null || echo 0)
    if (( carve_rc != 0 || got != this_bytes )); then
      echo "  Carve mismatch detected! Expected ${this_bytes}, got ${got}"
      rm -f "$chunk_path"
      printf '%s\t%s\t%s\n' "$repo_path" "$this_bytes" "-" >> "$MANIFEST"
      failed=$(( failed + 1 )); FAILED_LIST+=("${chunk_name} (carve)")
      file_ok=0
      continue
    fi
    # Carve verified, so it counts toward the carve rate even if the upload fails.
    carve_bytes=$(( carve_bytes + this_bytes )); carve_ms=$(( carve_ms + t1 - t0 ))
    printf '%s\t%s\n' "$ckey" "$local_sha" >> "$HASH_CACHE"
    printf '%s\t%s\t%s\n' "$repo_path" "$this_bytes" "$local_sha" >> "$MANIFEST"

    if upload_one "$chunk_path" "$repo_path"; then
      t2=$(now_ms)
      # Confirm the Hub stored exactly the bytes we carved.
      rsha=$(remote_sha "$repo_path")
      if [[ -n "$rsha" && "$rsha" != "$local_sha" ]]; then
        echo "  POST-UPLOAD HASH MISMATCH on ${repo_path}"
        failed=$(( failed + 1 )); FAILED_LIST+=("${chunk_name} (hash)")
        hash_bad=$(( hash_bad + 1 )); file_ok=0
      else
        uploaded=$(( uploaded + 1 ))
        [[ -n "$rsha" ]] && hash_ok=$(( hash_ok + 1 )) || size_only=$(( size_only + 1 ))
        up_bytes=$(( up_bytes + this_bytes )); up_ms=$(( up_ms + t2 - t1 ))
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "$(date -Is)" "$repo_path" "$this_bytes" "$(( t1 - t0 ))" "$(( t2 - t1 ))" >> "$TIMING_LOG"
      fi
    else
      failed=$(( failed + 1 )); FAILED_LIST+=("$chunk_name")
      file_ok=0
    fi
    rm -f "$chunk_path"
  done

  # SIDECAR: only once every part of this file is on the Hub, so its presence
  # doubles as a marker that the set is complete.
  if (( file_ok == 1 && WHOLE_HASH == 1 )) && [[ -n "$whole_sha" && -z "${REMOTE[$sidecar]:-}" ]]; then
    SC_TMP=$(mktemp)
    printf '%s  %s\n' "$whole_sha" "$base" > "$SC_TMP"
    if upload_one "$SC_TMP" "$sidecar"; then
      echo "  sidecar uploaded: ${sidecar}"
      sidecars=$(( sidecars + 1 ))
    else
      echo "  sidecar upload FAILED: ${sidecar}"
    fi
    rm -f "$SC_TMP"
  fi

  # A file is only useful once every part is present, so score it as a whole.
  if (( file_ok == 1 )); then
    complete=$(( complete + 1 ))
  else
    incomplete=$(( incomplete + 1 )); INCOMPLETE_LIST+=("$base")
  fi
done

# SUMMARY
separator
echo "FILES"
echo "  Complete   : ${complete}"
echo "  Incomplete : ${incomplete} (missing at least one part - re-run to finish)"
echo "  Skipped    : ${skipped} (missing files)"
if (( incomplete > 0 )); then
  printf '    - %s\n' "${INCOMPLETE_LIST[@]}"
fi

echo
echo "PARTS"
echo "  Uploaded this run : ${uploaded}"
echo "  Already present   : ${resumed}"
echo "  Failed            : ${failed}"
if (( failed > 0 )); then
  printf '    - %s\n' "${FAILED_LIST[@]}"
fi

echo
echo "VERIFICATION"
echo "  SHA-256 confirmed : ${hash_ok}"
echo "  Size only         : ${size_only} (no local hash; set VERIFY_HASH=1 to force)"
echo "  Hash mismatches   : ${hash_bad}"
echo "  Sidecars uploaded : ${sidecars}"

echo
echo "THROUGHPUT (this run only)"
if (( up_bytes == 0 || up_ms == 0 )); then
  echo "  No new bytes uploaded - no rate measured."
else
  awk -v cb="$carve_bytes" -v cms="$carve_ms" \
      -v ub="$up_bytes"    -v ums="$up_ms" -v tgt="$TARGET_GIB" '
  BEGIN {
    target = tgt * 1024 * 1024 * 1024
    carve_h = 0
    if (cb > 0 && cms > 0) {
      crate = cb / (cms / 1000)                    # bytes/s
      printf "  Carve  : %.1f MB/s  (%.2f GiB in %.0f s)\n",
             crate / 1e6, cb / 1073741824, cms / 1000
      carve_h = target / crate / 3600
    }
    urate = ub / (ums / 1000)                      # bytes/s
    umbps = urate * 8 / 1e6
    printf "  Upload : %.1f Mbps  (%.2f GiB in %.0f s)\n",
           umbps, ub / 1073741824, ums / 1000
    upload_h = target / urate / 3600
    printf "\n  Projection for %d GiB:\n", tgt
    if (carve_h > 0) printf "    carve  : %.1f h\n", carve_h
    printf "    upload : %.1f h\n", upload_h
    printf "    total  : %.1f h\n", carve_h + upload_h
    # Sanity check: a rate far above a domestic uplink means Xet deduplicated
    # the content and almost nothing crossed the wire.
    if (umbps > 300)
      printf "\n  WARNING: %.0f Mbps is implausible for a home uplink.\n         Check the \"New Data Upload\" line - if it reads kB, Xet\n         deduplicated this content and the rate is not a measurement.\n", umbps
  }'
fi
echo "  Per-part timings appended to ${TIMING_LOG}"

separator
echo "Verifying against ${REPO} ..."
python3 - "$REPO" "$MANIFEST" <<'PY'
import sys
from huggingface_hub import HfApi

repo, manifest = sys.argv[1], sys.argv[2]
try:
    info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
except Exception as e:
    print(f"  could not reach hub: {e}")
    sys.exit(1)

remote = {}
for s in info.siblings:
    remote[s.rfilename] = (s.size, s.lfs.sha256 if s.lfs and s.lfs.sha256 else None)

bad = 0
checked_hash = 0
for line in open(manifest):
    line = line.rstrip("\n")
    if not line:
        continue
    name, size, sha = line.split("\t")
    size = int(size)
    entry = remote.get(name)
    if entry is None:
        print(f"  MISSING on hub : {name}")
        bad += 1
        continue
    r_size, r_sha = entry
    if r_size != size:
        print(f"  SIZE MISMATCH  : {name}  local={size}  remote={r_size}")
        bad += 1
    elif sha != "-" and r_sha and sha != r_sha:
        print(f"  HASH MISMATCH  : {name}")
        bad += 1
    elif sha != "-" and r_sha:
        checked_hash += 1

if bad == 0:
    print(f"  all expected parts present; {checked_hash} verified by SHA-256.")
else:
    print(f"  {bad} problem(s) found.")
PY

rm -f "$MANIFEST"