#!/usr/bin/env python

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.cuda as tcuda
import torch.utils.data.distributed
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS,
                         RunningMetric, validate)


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    save_dir = os.path.join(args.base_path, config["data"]["data_root"])
    out_dir = os.path.join(args.out_dir, os.path.basename(args.config_file).split('.')[0])
    # assert hasattr(
    #     custom_dataset, config["data"]["train_dataset"]
    # ), f"{config['data']['train_dataset']} not a custom dataset"
    if 'undistort_f' in config["data"].keys():
        out_dir += f"_f{config['data']['undistort_f']}"
        valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
            test_mode=True, base_path=save_dir, crop=config["data"]["crop"],
            tgt_f=config["data"]["tgt_f"],
            undistort_f=config["data"]["undistort_f"],
            resize_im=config["data"]["resize_im"],
        )
    else:
        valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
            test_mode=True, base_path=save_dir, crop=config["data"]["crop"],
        )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"] if args.val_batch_sz is None else args.val_batch_sz,
        num_workers=4,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    is_normals = config["model"]["output_dim"] > 1
    if is_normals:
        metrics_tracker = RunningMetric(list(DICT_METRICS_NORMALS.keys()))
    else:
        metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    print("Start validation...")
    with torch.no_grad():
        validate.best_loss = np.inf
        validate(
            model,
            test_loader=valid_loader,
            config=config,
            metrics_tracker=metrics_tracker,
            context=context,
            scale_factor=args.scale_fac,
            save_dir=save_dir,
            out_dir=out_dir,
            vis=args.vis,
        )


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--base-path", default=os.environ.get("TMPDIR", ""))
    parser.add_argument("--scale-fac", type=float, default=1.0)
    parser.add_argument("--val-batch-sz", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default='show_dirs')
    parser.add_argument("--vis", action="store_true")


    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)
