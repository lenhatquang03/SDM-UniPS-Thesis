# Data transfer scripts

Shell utilities that move the training datasets (hdlong-complexv1, MerlMix,
PolarPS) from cloud storage onto a training machine. Nothing in `train.py`
calls them.

## What to run: step 3 only

All archives have already been uploaded to the Hugging Face dataset repo
`HUST-CVLab-PS/UniPS` with step 2 — the repo step 3 reads from. On a new
machine, **run only step 3**. Steps 1, 1.5 and 2 are kept for the record and
for re-uploading.

```bash
cd sdm_unips/data
pip install -U huggingface_hub              # provides the `hf` CLI
sudo apt-get install -y libarchive-tools    # bsdtar, for streaming extraction
hf auth login                               # only if the repo is private or gated

nohup ./step3_download_and_extract_HG_data.sh /path/to/datasets > step3.log 2>&1 &
tail -f step3.log
```

With no archive names it downloads and extracts **every** archive in the repo
into `/path/to/datasets`. Then point training at the extracted scene roots —
the directories whose children are scene folders (`light_means.config` inside
for hdlong and MerlMix, `normal.exr` for PolarPS):

```bash
python sdm_unips/train.py ... \
  --hdlong_dir <hdlong-complexv1 root> <MerlMix root> \
  --polarps_dir <PolarPS root>
```

Keep the `--hdlong_dir` order used by earlier runs (hdlong-complexv1 first,
then MerlMix): root order is part of the scene split's identity, and
`--resume` aborts on a change. See CLAUDE.md, *Several roots per source*.

## Overview

| Step | Script | Purpose | Status |
|---|---|---|---|
| 1 | `step1_rclone_download.sh` | Pull zip archives from the rclone remote | Done |
| 1.5 | `step1_5_extract_and_cleanup.sh` | Extract local zips, deleting each once verified | Optional |
| 2 | `step2_chunk_and_upload_HG_data.sh` | Split zips into ≤45 GiB parts and upload them to Hugging Face | **Done** |
| 3 | `step3_download_and_extract_HG_data.sh` | Download parts from Hugging Face, verify, extract, clean up | **Run this** |

Every script writes its timing log (`*.tsv`) to the **current directory**, so
run them from a directory you don't mind accumulating logs in.

---

## Step 1 — `step1_rclone_download.sh`

Copies zip archives from the rclone remote `thesis_data:DataThesis/zipped` to
`/workspace/data/zipped`, verifying each by checksum.

**Requires** `rclone` with a remote named `thesis_data` configured
(`rclone config`), and an existing `/workspace/data_scripts/` directory: the
script writes its filter file and detailed log there.

**Hard-coded.** Source, destination and file list are set at the top of the
script. Only `MerlMix/merlmix.zip` and `PolarPS/obj_ax.zip` are included; edit
the filter heredoc to fetch others.

```bash
nohup ./step1_rclone_download.sh >> rclone.log 2>&1 &
```

**Output.** The detailed log goes to `/workspace/data_scripts/download_<timestamp>.log`.
Exit status 0 means download and checksum verification succeeded.

**Quirk.** The failure message prints `$EXIT_CODE`, which is never set, so the
code shows blank. The script's own exit status carries the real rclone code.

---

## Step 1.5 — `step1_5_extract_and_cleanup.sh`

Extracts zip archives that are already on local disk, one at a time, and
deletes each archive only after its extraction is verified. Because it is
sequential, peak disk use is one archive plus its output.

For each archive:

1. Read the zip's central directory for the file count and uncompressed size.
2. **Skip** it if free space is below the uncompressed size + 1 GiB.
3. **CRC-test** it (`unzip -t`); a corrupt archive is left in place.
4. Extract, printing progress every 30 s.
5. **Verify** every file listed in the archive exists on disk.
6. Append a row to `./unzip_timings.tsv`, then **delete** the zip.

An archive that is skipped or fails at any step is never deleted.

**Not executable in the repo** — run it through `bash` (or `chmod +x` first):

```bash
bash step1_5_extract_and_cleanup.sh <dest_dir> <archive.zip> [archive.zip ...]

# dry run: extract but keep the zip
KEEP_ZIP=1 bash step1_5_extract_and_cleanup.sh /workspace/data /workspace/data/zipped/MerlMix/merlmix.zip
```

| Variable | Effect |
|---|---|
| `SKIP_TEST=1` | Skip the CRC pre-check (saves one full read per archive) |
| `KEEP_ZIP=1` | Extract but do not delete |
| `VERBOSE=1` | List every extracted file |

**Don't run it on zips that still need uploading with step 2** — it deletes
them (unless `KEEP_ZIP=1`).

---

## Step 2 — `step2_chunk_and_upload_HG_data.sh` (already done)

Uploads archives to a Hugging Face dataset repo. Hugging Face rejects files
above 50 GiB, so larger archives are split first.

- A file no larger than the chunk size is uploaded whole.
- A larger file is carved with `dd` into `<name>.part00`, `<name>.part01`, …,
  **one part at a time**. Each part is hashed while it is carved, uploaded,
  checked against the SHA-256 the Hub reports, then deleted locally. Peak disk
  use is one part; a part is refused if free space is below its size + 1 GiB.
- Parts go into a repo subfolder chosen by filename: `obj_*` → `polarps/`,
  `hdlong_*` → `hdlong/`, `merlmix*` → `merlmix/`, anything else → the source
  file's parent directory name.
- Once every part of a file is up, a `<name>.sha256` sidecar holding the
  whole-file hash is uploaded. Step 3 uses it to verify the reassembled archive.
- At the end, every expected part is checked against the Hub (presence, size,
  SHA-256).

**Resumable.** Re-running skips parts already on the Hub: hash-verified when a
local hash is cached (or with `VERIFY_HASH=1`), otherwise by size only.
**Always re-run with the same chunk size.** A different size re-splits
everything, and step 3 would stitch old and new parts together.

**Requires** a write-access token (`hf auth login`), the `hf` CLI, and
`python3` with `huggingface_hub`.

**Interactive.** It prompts for the repo (unless `HF_REPO` is set), the
absolute paths of the files to upload, and the chunk size in GiB (1–45). Run it
in a terminal or `tmux`, not under `nohup`.

```bash
HF_REPO=<org>/<repo> ./step2_chunk_and_upload_HG_data.sh
```

| Variable | Effect |
|---|---|
| `HF_REPO` | Target repo; skips that prompt |
| `VERIFY_HASH=1` | On resume, re-read the source to hash parts with no cached hash |
| `WHOLE_HASH=0` | Don't compute or upload `.sha256` sidecars |

**Local files:** `./hf_upload_timings.tsv` (per-part timings) and
`./hf_upload_hashes.tsv` (hash cache used on resume).

---

## Step 3 — `step3_download_and_extract_HG_data.sh`

Downloads archives from `HUST-CVLab-PS/UniPS` (set in `REPOS` at the top of
the script), verifies them, extracts them into one directory, and deletes
everything it downloaded.

```bash
./step3_download_and_extract_HG_data.sh <download_dir> [archive_name ...]
./step3_download_and_extract_HG_data.sh                  # prompts for both
```

- `<download_dir>` is where archives are extracted. Parts are staged in
  `<download_dir>/.parts/`, which is removed once empty.
- `archive_name` is the **original** file name, not a part name (e.g.
  `merlmix.zip`). With no names, every archive in the repo is processed and the
  list is printed first. To see what's in the repo without downloading:

  ```bash
  python3 -c "from huggingface_hub import list_repo_files as f; print('\n'.join(f('HUST-CVLab-PS/UniPS', repo_type='dataset')))"
  ```

For each archive:

1. Collect its parts in order. It **refuses** to extract if a part number is
   missing, and notes when the last part is full-size (a further part may be missing).
2. Fetch the `.sha256` sidecar if one exists.
3. Print a rough space estimate. This is a warning only; it carries on regardless.
4. Download and extract in one of two modes:

| Mode | How it works | Extra disk |
|---|---|---|
| `STREAM=1` (default) | Each part: download → check size → check SHA-256 against the Hub → pipe into `bsdtar` → delete. The whole stream is hashed on the way and compared with the sidecar at the end. | ~one part + extracted output |
| `STREAM=0` | Each part: download → check size and SHA-256 → append to a merged zip → delete. Then verify the merged zip against the sidecar, `unzip`, delete the merged zip. | ~all parts + extracted output |

`STREAM=1` needs `bsdtar` (package `libarchive-tools`). If it is missing, the
script offers to install it with apt. Set `AUTO_APT=1` to install without
asking; this matters under `nohup`, where it can't prompt and **silently falls
back to `STREAM=0`**.

| Variable | Effect |
|---|---|
| `STREAM=0` | Merge-then-unzip instead of streaming |
| `AUTO_APT=1` | Install `libarchive-tools` without prompting |
| `VERBOSE_EXTRACT=1` | List every extracted file |

**Log:** `./hf_download_timings.tsv` (per-part timings). The final summary
reports archives extracted / failed / not found and download throughput.

### If an archive fails

- **`STREAM=1`:** extraction happens as parts arrive, so a failure on a later
  part (download, size or hash) leaves a **partially extracted** tree from the
  earlier parts, and those parts are already deleted. Delete the partial output
  and re-run for that archive only, e.g.
  `./step3_download_and_extract_HG_data.sh /path/to/datasets merlmix.zip`.
  On an unreliable connection, use `STREAM=0`, which extracts only after every
  part has passed. A `WHOLE-FILE HASH MISMATCH` is reported *after* extraction,
  so don't trust that archive's output.
- **`STREAM=0`:** the merged zip is kept in `<download_dir>/.parts/` when
  verification passes but `unzip` fails. To retry without downloading again,
  run `unzip -o <download_dir>/.parts/<archive_name> -d <download_dir>` by
  hand. **Re-running step 3 truncates that file** and downloads everything again.
- There is no resume: re-running an archive downloads it from scratch.
- If `.parts/` is left behind holding only `.cache/huggingface/`, that is the
  download client's metadata and is safe to delete.
