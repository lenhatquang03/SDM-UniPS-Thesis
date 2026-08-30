"""Off-box backup of the minimum needed to resume a run, to a HuggingFace repo.

A rented GPU instance can be destroyed with no warning, and everything on it
goes with it. Almost all of that is regenerable for free -- the datasets come
back from HuggingFace, the code from git, `scene_manifest.json` from a rescan
(the manifest and its fingerprint are relative to the dataset roots, so a fresh
scan of the same data reproduces the same pool), and `normal.pytmodel` is
re-exported from `best.pt`. The trained weights are the one thing that cannot
be reconstructed at any price, so they are the one thing this module ships off
the box.

What gets uploaded, and why each file is on the list:

- `latest.pt`   -- the epoch checkpoint just written. A FULL checkpoint
                   (model + both AdamW moments + scheduler position + every RNG
                   stream + epoch + `best_val_loss` + patience counter), so
                   `--resume <path>/latest.pt` on a fresh box continues the run
                   properly instead of warm-starting from bare weights.
- `best.pt`     -- the deliverable. Not needed to resume, but if the run never
                   improves again after a crash there would otherwise be no
                   selected model to show for it.
- `train.jsonl` -- the loss/MAE curves. Small, and the thing the thesis plots.
- `config*.json`-- the frozen argument snapshot, i.e. how to relaunch.

Deliberately NOT uploaded: `step_*.pt` (resume granularity is one epoch, so a
mid-epoch snapshot buys nothing once `latest.pt` exists), `normal.pytmodel`
(re-exportable from `best.pt`), `scene_manifest.json` (regenerable, and large).

Design constraints this module is built to respect:

- **It must never kill a run.** Every failure path is swallowed with a printed
  warning. An HF outage at epoch 150 must not cost epoch 151.
- **It must not be an import-time dependency.** `huggingface_hub` is imported
  inside the call, so a box without it trains normally.
- **It must not race the checkpoint writer.** The caller invokes it only after
  `trainer.save` has renamed its `*.tmp` into place and `prune_checkpoints` has
  run, so every file it reads is complete and none is about to be deleted.
- **Fixed remote filenames, overwritten.** HuggingFace keeps LFS history, so
  uniquely-named checkpoints would accumulate the full run's worth of weights
  in the repo. `--hf_backup_every` controls how many revisions that history
  ends up holding; `HfApi().super_squash_history()` collapses it when it grows.
"""

import glob
import os
import time


class HFBackup:
    """Uploads the resume-critical files to `sdm-ckpt/<model_name>/` in a repo.

    Disabled instances are inert: `maybe_upload` returns immediately, so the
    no-flag code path is identical to having no backup at all.
    """

    def __init__(self, repo_id, model_name, repo_type='dataset', every=1,
                 enabled=False):
        self.repo_id = repo_id
        self.model_name = model_name
        self.repo_type = repo_type
        self.every = max(0, int(every))
        self.enabled = bool(enabled)
        self.prefix = f'sdm-ckpt/{model_name}'
        # remote name -> (size, mtime) of the copy already pushed, so an
        # unchanged `best.pt` is not re-uploaded on every backup point.
        self._sent = {}

    # -- preflight ---------------------------------------------------------
    def preflight(self):
        """Check credentials and repo access once, at startup.

        Without this the first upload attempt happens at the first backup
        point, which on a long run can be hours in -- exactly the wrong time to
        discover the token is missing. Never raises: a broken backup is a
        reason to warn, not to refuse to train.
        """
        if not self.enabled:
            return False
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            who = api.whoami()
            api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type)
        except Exception as exc:
            print(f'[HF-BACKUP] preflight FAILED: {type(exc).__name__}: {exc}')
            print(f'[HF-BACKUP] uploads will be attempted anyway and may fail; '
                  f'training is unaffected.')
            return False
        name = who.get('name', '?') if isinstance(who, dict) else '?'
        every = f'every {self.every} epoch(s)' if self.every else 'on demand'
        print(f'[HF-BACKUP] enabled | user={name} | '
              f'{self.repo_type}:{self.repo_id}/{self.prefix} | {every}')
        return True

    # -- upload ------------------------------------------------------------
    def maybe_upload(self, epoch, resume_path, ckpt_dir, log_dir, force=False):
        """Upload at an epoch boundary, subject to `--hf_backup_every`.

        `force` bypasses the interval (used for the final checkpoint, which
        must go up regardless of where the run happened to stop).
        """
        if not self.enabled:
            return False
        if not force and self.every > 0 and (epoch + 1) % self.every != 0:
            return False
        if not force and self.every == 0:
            return False
        try:
            return self._upload(epoch, resume_path, ckpt_dir, log_dir, force)
        except Exception as exc:
            # Deliberately broad: nothing this module can fail at is worth
            # ending a multi-day run over.
            print(f'[HF-BACKUP] upload failed at epoch {epoch}: '
                  f'{type(exc).__name__}: {exc}')
            print('[HF-BACKUP] training continues; local checkpoints are intact.')
            return False

    def _upload(self, epoch, resume_path, ckpt_dir, log_dir, force):
        from huggingface_hub import CommitOperationAdd, HfApi

        ops, staged, total = [], [], 0

        def add(local, remote):
            """Stage `local` unless the identical file was already pushed."""
            nonlocal total
            if not local or not os.path.isfile(local):
                return
            st = os.stat(local)
            key = (st.st_size, int(st.st_mtime))
            if self._sent.get(remote) == key:
                return
            ops.append(CommitOperationAdd(
                path_in_repo=f'{self.prefix}/{remote}', path_or_fileobj=local))
            staged.append((remote, key, st.st_size))
            total += st.st_size

        add(resume_path, 'latest.pt')
        add(os.path.join(ckpt_dir, 'best.pt'), 'best.pt')
        add(os.path.join(log_dir, 'train.jsonl'), 'train.jsonl')
        add(os.path.join(log_dir, 'eval.jsonl'), 'eval.jsonl')
        for cfg in sorted(glob.glob(os.path.join(log_dir, 'config*.json'))):
            add(cfg, os.path.basename(cfg))

        if not ops:
            return False

        names = ', '.join(r for r, _, _ in staged)
        print(f'[HF-BACKUP] epoch {epoch}: uploading {len(ops)} file(s), '
              f'{total / 2**30:.2f} GiB ({names}) ...')
        t0 = time.time()
        HfApi().create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=ops,
            commit_message=f'{self.model_name}: epoch {epoch}'
                           + (' (final)' if force else ''),
        )
        # Only after the commit succeeds, so a failed upload is retried at the
        # next backup point rather than being recorded as sent.
        for remote, key, _ in staged:
            self._sent[remote] = key
        dt = time.time() - t0
        rate = (total / 2**20) / max(dt, 1e-6)
        print(f'[HF-BACKUP] epoch {epoch}: done in {dt:.1f}s '
              f'({rate:.1f} MiB/s) -> {self.repo_id}/{self.prefix}')
        return True
