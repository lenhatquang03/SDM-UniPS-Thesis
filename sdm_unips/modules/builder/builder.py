"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023).

Inference orchestration for surface-normal recovery.
"""


import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from modules.model import model, decompose_tensors
from modules.model.model_utils import *
from modules.utils import compute_mae


class builder():
    def __init__(self, args, device):
        self.device = device
        self.args = args
        model_dir = f'{args.checkpoint}/normal'
        self.net_nml = model.Net(args.pixel_samples, device).to(self.device)
        self.net_nml = torch.nn.DataParallel(self.net_nml)
        self.net_nml = self.load_models(self.net_nml, model_dir)
        self.net_nml.module.no_grad()

        print(f"canonical resolution: {self.args.canonical_resolution} x {self.args.canonical_resolution}  ")
        print(f"pixel samples: {self.args.pixel_samples}\n")

    def separate_batch(self, batch):
        I = batch[0].to(self.device)
        N = batch[1].to(self.device)
        M = batch[2].to(self.device)
        nImgArray = batch[3]
        roi = batch[4]
        return I, N, M, nImgArray, roi

    def load_models(self, model, dirpath):
        """Load the single `*.pytmodel` under `dirpath` into `model`.

        Two silent-failure modes are closed here, both of which produced a
        *randomly initialised* network that still printed a success message and
        went on to report ~76 deg MAE on DiLiGenT:

        1. **`module.` prefix mismatch.** `model` is DataParallel-wrapped by the
           caller, so its keys carry a `module.` prefix. Upstream's released
           weights were saved from a wrapped model and match; `Trainer` saves
           `self.net.state_dict()` from a *bare* `Net` (trainer.py:158, :947)
           and does not. Under `strict=False` every key is then simultaneously
           "missing" and "unexpected", zero parameters transfer, and nothing is
           raised. The prefix is normalised in whichever direction is needed.
        2. **Ambiguous glob.** The original `"".join(glob.glob(...))` silently
           concatenated the paths when the directory held more than one
           `.pytmodel`, yielding a nonexistent filename, which `loadmodel`
           reports as "not Found" before returning the model untouched.

        Anything short of a full key match now raises.
        """
        candidates = sorted(glob.glob(f'{dirpath}/*.pytmodel'))
        if len(candidates) != 1:
            raise RuntimeError(
                f'Expected exactly one *.pytmodel in {dirpath}, found '
                f'{len(candidates)}: {[os.path.basename(c) for c in candidates]}. '
                f'`load_models` must be pointed at a directory holding only the '
                f'weights to load.')
        pytmodel = candidates[0]

        params = torch.load(pytmodel, map_location=self.device)
        # A training checkpoint nests the weights; a *.pytmodel is already bare.
        if isinstance(params, dict) and 'model' in params and 'optimizer' in params:
            params = params['model']

        wants_prefix = any(k.startswith('module.') for k in model.state_dict())
        has_prefix = any(k.startswith('module.') for k in params)
        if wants_prefix and not has_prefix:
            params = {f'module.{k}': v for k, v in params.items()}
            print('[builder] Added the "module." prefix to the checkpoint keys '
                  '(saved from an unwrapped model).')
        elif has_prefix and not wants_prefix:
            params = {k[len('module.'):]: v for k, v in params.items()}
            print('[builder] Stripped the "module." prefix from the checkpoint '
                  'keys (saved from a DataParallel model).')

        missing, unexpected = model.load_state_dict(params, strict=False)
        n_total = len(model.state_dict())
        n_loaded = n_total - len(missing)
        if n_loaded == 0:
            raise RuntimeError(
                f'{pytmodel}: none of its {len(params)} keys matched the model '
                f'({n_total} parameters). The network would be randomly '
                f'initialised. First checkpoint key: '
                f'{next(iter(params), "<empty>")!r}; first model key: '
                f'{next(iter(model.state_dict()), "<empty>")!r}.')
        if missing or unexpected:
            raise RuntimeError(
                f'{pytmodel}: partial load -- {n_loaded}/{n_total} parameters '
                f'matched, {len(missing)} missing, {len(unexpected)} unexpected. '
                f'Refusing to run on a half-initialised network. '
                f'Missing e.g. {missing[:3]}; unexpected e.g. {unexpected[:3]}.')

        print(f'Loading pretrained model... {pytmodel} '
              f'({n_loaded}/{n_total} parameters)')
        return model

    def run(self,
            canonical_resolution=None,
            testdata=None,
            max_image_resolution=None,
            ):
        """Recover normal maps for every object in `testdata`.

        Returns one record per object, in iteration order:
            {'objname', 'normal', 'normal_gt', 'error_map', 'mae'}
        `normal` is the full-canvas float32 prediction in RGB, `error_map` the
        per-pixel angular error in **degrees, unclamped** (the `error.png`
        written below is clamped to 90 and rescaled for 8-bit display), and
        `normal_gt` / `error_map` / `mae` are None when no ground truth was
        found. The return value is additive -- `main.py` ignores it.
        """

        testdata.max_image_resolution = max_image_resolution
        test_data_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        results = []
        for batch_test in test_data_loader:
            I, N, M, nImgArray, roi = self.separate_batch(batch_test)
            roi = roi[0].numpy()
            h_ = roi[0]
            w_ = roi[1]
            r_s = roi[2]
            r_e = roi[3]
            c_s = roi[4]
            c_e = roi[5]
            B, C, H, W, Nimg = I.shape

            with torch.no_grad():
                if self.args.scalable:
                    patch_size = 512
                    patches_I = decompose_tensors.divide_tensor_spatial(I.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W), block_size=patch_size, method='tile_stride')
                    patches_I = patches_I.reshape(B, Nimg, -1, C, patch_size, patch_size).permute(0, 2, 3, 4, 5, 1)
                    sliding_blocks = patches_I.shape[1]
                    patches_M = decompose_tensors.divide_tensor_spatial(M, block_size=patch_size, method='tile_stride')

                    patches_nml = []
                    for k in range(sliding_blocks):
                        print(f"Recovering normal map(s): {k+1} / {sliding_blocks}")
                        if torch.sum(patches_M[:, k, :, :, :]) > 0:
                            pI = patches_I[:, k, :, :, :, :]
                            pI = F.interpolate(pI.permute(0, 4, 1, 2, 3).reshape(-1, pI.shape[1], pI.shape[2], pI.shape[3]), size=(patch_size, patch_size), mode='bilinear', align_corners=True).reshape(B, Nimg, C, patch_size, patch_size).permute(0, 2, 3, 4, 1)
                            pM = F.interpolate(patches_M[:, k, :, :, :], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                            nout = self.net_nml(pI, pM, nImgArray.reshape(-1, 1), decoder_resolution=patch_size * torch.ones(pI.shape[0], 1), canonical_resolution=canonical_resolution * torch.ones(pI.shape[0], 1))
                            nout = (F.interpolate(nout, size=(patch_size, patch_size), mode='bilinear', align_corners=True) * pM).cpu()
                            patches_nml.append(nout)
                        else:
                            patches_nml.append(torch.zeros((B, 3, patch_size, patch_size)))
                    patches_nml = torch.stack(patches_nml, dim=1)
                    merged_tensor_nml = decompose_tensors.merge_tensor_spatial(patches_nml.permute(1, 0, 2, 3, 4), method='tile_stride')
                    nml = merged_tensor_nml.squeeze().permute(1, 2, 0)
                else:
                    print("Recovering normal map(s) 1 / 1")
                    nout = self.net_nml(I, M, nImgArray.reshape(-1, 1), decoder_resolution=testdata.data.h * torch.ones(I.shape[0], 1), canonical_resolution=canonical_resolution * torch.ones(I.shape[0], 1))
                    nml = (nout * M).squeeze().permute(1, 2, 0).cpu().detach()
                    del nout

                nml = nml.cpu().numpy()
                nml = cv2.resize(nml, dsize=(c_e - c_s, r_e - r_s), interpolation=cv2.INTER_CUBIC)
                mask = np.float32(np.abs(1 - np.sqrt(np.sum(nml * nml, axis=2))) < 0.5)
                nml = np.divide(nml, np.linalg.norm(nml, axis=2, keepdims=True) + 1.0e-12)
                nml = nml * mask[:, :, np.newaxis]
                nout = np.zeros((h_, w_, 3), np.float32)
                nout[r_s:r_e, c_s:c_e, :] = nml

                record = {'objname': testdata.data.objname, 'normal': nout,
                          'normal_gt': None, 'error_map': None, 'mae': None}

                if torch.sum(N) > 0:
                    n_true = N.permute(0, 2, 3, 1).squeeze().cpu().numpy()
                    mask = np.float32(np.abs(1 - np.sqrt(np.sum(n_true * n_true, axis=2))) < 0.5)
                    mae, emap = compute_mae.compute_mae_np(nout, n_true, mask=mask)
                    print(f"Mean Angular Error (MAE) is {mae:.3f}\n")
                    emap = emap.squeeze()
                    # Capture before the display clamp below rewrites emap in place.
                    record['normal_gt'] = n_true
                    record['error_map'] = emap.copy()
                    record['mae'] = float(mae)
                    thresh = 90
                    emap[emap >= thresh] = thresh
                    emap = emap / thresh
                    cv2.imwrite(f'{testdata.data.data_workspace}/error.png', 255 * emap)

                cv2.imwrite(f'{testdata.data.data_workspace}/normal.png', 255 * (0.5 * (1 + nout[:, :, ::-1])))
                results.append(record)

        return results
