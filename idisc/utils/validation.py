import json
import os
from typing import Any, Dict, Optional

import numpy as np
import tqdm
import torch
import torch.utils.data.distributed
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from idisc.utils.metrics import RunningMetric
from idisc.utils.misc import is_main_process
from idisc.utils.visualization import save_val_imgs, save_file_ply
from idisc.utils.unproj_pcd import reconstruct_pcd, reconstruct_pcd_fisheye


def log_losses(losses_all):
    for loss_name, loss_val in losses_all.items():
        print(f"Test/{loss_name}: ", loss_val)


def update_best(metrics_all, metrics_best="abs_rel"):
    curr_loss = []
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)

    curr_loss = np.mean(curr_loss)
    if curr_loss < validate.best_loss:
        validate.best_loss = curr_loss
        validate.best_metrics = metrics_all

    for metrics_name, metrics_value in metrics_all.items():
        try:
            print(
                f"{metrics_name} {round(validate.best_metrics[metrics_name], 4)} ({round(metrics_value, 4)})"
            )
        except:
            print(f"Error in best. {metrics_name} ({round(metrics_value, 4)})")


def save_model(
    metrics_all, state_dict, run_save_dir, step, config, metrics_best="abs_rel"
):
    curr_loss = []
    curr_dataset = config["data"]["train_dataset"]
    for metrics_name, metrics_value in metrics_all.items():
        if metrics_best in metrics_name:
            curr_loss.append(metrics_value)
    curr_loss = np.mean(curr_loss)

    if curr_loss == validate.best_loss:
        if step > 15000:
            try:
                torch.save(
                    state_dict, os.path.join(run_save_dir, f"{curr_dataset}-best.pt")
                )
                with open(
                    os.path.join(run_save_dir, f"{curr_dataset}-config.json"), "w+"
                ) as fp:
                    json.dump(config, fp)
            except OSError as e:
                print(f"Error while saving model: {e}")
            except:
                print("Generic error while saving")


def validate(
    model: nn.Module,
    test_loader: DataLoader,
    metrics_tracker: RunningMetric,
    context: torch.autocast,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    save_dir: Optional[str] = None,
    step: int = 0,
    scale_factor: float = 1.0,
    vis: bool = False,
    out_dir: Optional[str] = None,
):
    ds_losses = {}
    device = model.device

    for i, batch in enumerate(test_loader):
        print(f'Processing {i} / {len(test_loader)} batches')
        with context:
            gt, mask = batch["gt"].to(device), batch["mask"].to(device)
            preds, losses, _ = model(batch["image"].to(device), gt, mask)

        losses = {k: v for l in losses.values() for k, v in l.items()}
        for loss_name, loss_val in losses.items():
            ds_losses[loss_name] = (
                loss_val.detach().cpu().item() + i * ds_losses.get(loss_name, 0.0)
            ) / (i + 1)

        if 'info' in batch.keys() and 'pred_scaler' in batch['info'].keys():
            scale_factor = batch['info']['pred_scaler'].to(device)
            preds *= scale_factor[:, None, None, None]
        else:
            preds *= scale_factor
        
        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1), preds.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1)
        )
        
        if vis and i % 10 == 0:
            save_img_dir = os.path.join(out_dir, 'val_imgs')
            os.makedirs(save_img_dir, exist_ok=True)
            rgb = save_val_imgs(
                i,
                preds[0],
                gt[0],
                batch["image"][0],
                f'rgb_{i:06d}_merge.jpg',
                save_img_dir
            )
            
            # pcd
            pred_depth = preds[0, 0].detach().cpu().numpy()
            if config['data']['data_root'] == 'kitti360' and 'undistort_f' not in config['data'].keys():
                fisheye_file = batch['info']['image_filename'][0]
                if 'image_02' in fisheye_file:
                    grid_fisheye = np.load(os.path.join(save_dir, 'fisheye', 'grid_fisheye_02.npy'))
                    mask_fisheye = np.load(os.path.join(save_dir, 'fisheye', 'mask_left_fisheye.npy'))
                elif 'image_03' in fisheye_file:
                    grid_fisheye = np.load(os.path.join(save_dir, 'fisheye', 'grid_fisheye_03.npy'))
                    mask_fisheye = np.load(os.path.join(save_dir, 'fisheye', 'mask_right_fisheye.npy'))
                grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (pred_depth.shape[1], pred_depth.shape[0]))
                mask_fisheye = cv2.resize(mask_fisheye.astype(np.uint8), (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                pcd = reconstruct_pcd_fisheye(pred_depth, grid_fisheye, mask=mask_fisheye)
            else:
                intrinsics = batch['info']['camera_intrinsics'][0].detach().cpu().numpy()
                pcd = reconstruct_pcd(pred_depth, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
            save_pcd_dir = os.path.join(out_dir, 'val_pcds')
            os.makedirs(os.path.join(save_pcd_dir), exist_ok=True)
            pc_file = os.path.join(save_pcd_dir, f'pcd_{i:06d}.ply')
            pcd = pcd.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            non_zero_indices = pcd[:, -1] > 0
            pcd = pcd[non_zero_indices]
            rgb = rgb[non_zero_indices]
            save_file_ply(pcd, rgb, pc_file)
            

    losses_all = ds_losses
    metrics_all = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()

    if is_main_process():
        log_losses(losses_all=losses_all)
        update_best(metrics_all=metrics_all, metrics_best="abs_rel")
        if out_dir is not None and run_id is not None:
            run_save_dir = os.path.join(out_dir, run_id)
            os.makedirs(run_save_dir, exist_ok=True)

            with open(os.path.join(run_save_dir, f"metrics_{step}.json"), "w") as f:
                json.dump({**losses_all, **metrics_all}, f)
            save_model(
                metrics_all=metrics_all,
                state_dict=model.state_dict(),
                config=config,
                metrics_best="abs_rel",
                run_save_dir=run_save_dir,
                step=step,
            )
