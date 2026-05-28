"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import glob
import torch.utils.data as data
from .dataloader import realdata
from .dataloader.mixed import build_mixed_split
from .dataloader.diligent import DiligentLoader
import numpy as np
import os

class dataio(data.Dataset):
    def __init__(self, mode, args):
        self.mode = mode
        data_root = args.test_dir
        extension = args.test_ext
        self.numberOfImageBuffer = args.max_image_num
        self.prefix= args.test_prefix
        self.mask_margin = args.mask_margin
        self.outdir = args.session_name
        self.data_root = data_root
        self.extension = extension
        self.data_name = []
        self.set_id = []
        self.valid = []
        self.sample_id = []
        self.dataCount = 0
        self.dataLength = -1
        self.mode = mode
        self.max_image_resolution = None

        print('Exploring %s' % (data_root))
        objlist = glob.glob(f"{data_root}/*{extension}")
        objlist = sorted(objlist)
        self.objlist = objlist
        print(f"Found {len(self.objlist)} objects!\n")
        self.data = realdata.dataloader(self.numberOfImageBuffer, mask_margin=self.mask_margin, outdir=self.outdir)

    def __getitem__(self, index_):

        objid = index_
        objdir = self.objlist[objid]
        self.data.load(objdir, prefix = self.prefix, max_image_resolution = self.max_image_resolution)
        img = self.data.I.transpose(2,0,1,3) # c, h, w, N
        numberOfImages = self.data.I.shape[3]
        nml = self.data.N.transpose(2,0,1) # 3, h, w
        mask = np.transpose(self.data.mask, (2,0,1)) # 1, h, w
        roi = self.data.roi
        return img, nml, mask, numberOfImages, roi

    def __len__(self):
        return len(self.objlist)


def build_train_dataset(args, augment=True):
    """Build the mixed (hdlong-complexv1 + PolarPS) training split.

    Returns only the train half of the deterministic scene-level split; the
    held-out val half is obtained via `build_val_dataset` with the same
    seed, guaranteeing the two are disjoint.
    """
    train_set, _ = build_mixed_split(args, augment=augment)
    return train_set


def build_val_dataset(args):
    """Build the held-out validation split (never augmented)."""
    _, val_set = build_mixed_split(args, augment=False)
    return val_set


class DiligentEvalDataset(data.Dataset):
    """DiLiGenT evaluation dataset, one item per scene + K-sweep trial.

    Index layout: i = obj_idx * (len(K_list) * trials_per_K) + k * trials_per_K + t.
    Each __getitem__ pre-samples K observations and returns the tensors needed
    by `Net.forward(training=False)` plus per-pixel GT for MAE.
    """

    def __init__(self, eval_dir, K_list=(2, 4, 8, 16, 32, 64, 96),
                 trials_per_K=10, side=512, seed=2024):
        if not os.path.isdir(eval_dir):
            raise RuntimeError(f'DiLiGenT eval root not found: {eval_dir}')
        # Find scene directories named "*PNG" — the standard DiLiGenT layout.
        candidates = sorted(glob.glob(os.path.join(eval_dir, '*PNG')))
        if not candidates:
            # Fallback: any sub-directory that has filenames.txt.
            candidates = sorted(d for d in glob.glob(os.path.join(eval_dir, '*'))
                                if os.path.isdir(d)
                                and os.path.isfile(os.path.join(d, 'filenames.txt')))
        if not candidates:
            raise RuntimeError(f'No DiLiGenT scenes under {eval_dir}')
        self.scenes = candidates
        self.K_list = tuple(int(k) for k in K_list)
        self.trials_per_K = int(trials_per_K)
        self.side = int(side)
        self.seed = int(seed)
        self._loaders = {}
        print(f'[DiligentEval] {len(self.scenes)} scenes  '
              f'K={list(self.K_list)}  trials_per_K={self.trials_per_K}')

    def __len__(self):
        return len(self.scenes) * len(self.K_list) * self.trials_per_K

    def _decompose(self, idx):
        per_obj = len(self.K_list) * self.trials_per_K
        obj_idx = idx // per_obj
        rest = idx % per_obj
        k_idx = rest // self.trials_per_K
        trial = rest % self.trials_per_K
        return obj_idx, k_idx, trial

    def get_meta(self, idx):
        obj_idx, k_idx, trial = self._decompose(idx)
        scene = self.scenes[obj_idx]
        return {
            'obj_idx': obj_idx,
            'objname': os.path.basename(scene),
            'K': self.K_list[k_idx],
            'trial': trial,
        }

    def __getitem__(self, idx):
        obj_idx, k_idx, trial = self._decompose(idx)
        scene = self.scenes[obj_idx]
        K = self.K_list[k_idx]
        loader = self._loaders.get(scene)
        if loader is None:
            loader = DiligentLoader(side=self.side)
            loader.load(scene)
            self._loaders[scene] = loader

        rng = np.random.RandomState(self.seed + 1009 * obj_idx + 17 * k_idx + trial)
        I, N, M, K_used = loader.sample(K, rng)
        # Pad to a fixed K_max so default collate can stack across batches if used.
        # (We typically run batch_size=1 in eval, so this is just safety.)
        I = I.transpose(2, 0, 1, 3)             # (3, h, w, K)
        N = N.transpose(2, 0, 1).astype(np.float32)
        M = M.transpose(2, 0, 1).astype(np.float32)
        return I, N, M, np.int64(K_used), self.K_list[k_idx], obj_idx, trial
