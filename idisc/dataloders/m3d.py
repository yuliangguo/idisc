"""

Author: Yuliang Guo

This is compatible for the Hypersim dataset resized and bordered to fit NYU focal and resolution

"""

import os

import numpy as np
import cv2
import torch
from PIL import Image

from .dataset import BaseDataset, resize_for_input

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class MatterPort3DDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [1 / np.tan(np.pi/512), 0.000000e00, 256.],
                [0.000000e00, 1 / np.tan(np.pi/512), 512.],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    min_depth = 0.01
    max_depth = 80
    test_split = "m3d_test.txt"
    train_split = "m3d_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=1,
        crop=None,
        benchmark=False,
        augmentations_db={},
        masked=False,
        normalize=True,
        tgt_f = 590.78,
        resize_ratio = 1.0,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.height = 512
        self.width = 1024
        self.tgt_f = tgt_f
        self.resize_ratio = resize_ratio
        self.height = int(self.height * self.resize_ratio)
        self.width = int(self.width * self.resize_ratio)
        self.masked = masked

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join(self.base_path, self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None":
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                img_info["camera_intrinsics"] = self.CAM_INTRINSIC['ALL'][:, :3].clone()
                
                # setup pred_scale_factor due to conversion to target focal length
                if self.tgt_f > 0:
                    img_info["pred_scale_factor"] = (img_info["camera_intrinsics"][0, 0] + img_info["camera_intrinsics"][1, 1]).item() / 2 / self.tgt_f
                    img_info["camera_intrinsics"][0, 0] /= img_info["pred_scale_factor"]
                    img_info["camera_intrinsics"][1, 1] /= img_info["pred_scale_factor"]
                else:
                    img_info["pred_scale_factor"] = 1.0
                
                self.dataset.append(img_info)
        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image = np.asarray(
            # Image.open(
            #     os.path.join(self.base_path, self.dataset[idx]["image_filename"])
            # )
            Image.open(self.dataset[idx]["image_filename"])
        )
        depth = (
            np.asarray(
                # Image.open(
                #     os.path.join(
                #         self.base_path, self.dataset[idx]["annotation_filename_depth"]
                #     )
                # )
                # Image.open(self.dataset[idx]["annotation_filename_depth"])
                cv2.imread(self.dataset[idx]["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        
        # resizing process to certain ratio
        if self.resize_ratio != 1.0:
            ori_h, ori_w, _ = image.shape
            fwd_sz = (int(ori_h * self.resize_ratio), int(ori_w * self.resize_ratio)) # online compute because kitti has inconsistent image size
            image, depth, pad, pred_scale_factor = resize_for_input(image, depth, fwd_sz, info["camera_intrinsics"], [ori_h, ori_w], 1.0)
            info['pred_scale_factor'] *= pred_scale_factor
            info['pad'] = pad
            if not self.test_mode:
                depth /= info['pred_scale_factor']
    
        image, gts, info = self.transform(image=image, gts={"depth": depth}, info=info)
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}
        
    def get_pointcloud_mask(self, shape):
        mask = np.zeros(shape)
        height_start, height_end = 45, self.height - 9
        width_start, width_end = 41, self.width - 39
        mask[height_start:height_end, width_start:width_end] = 1
        return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.height
        width_start, width_end = 0, self.width
        image = image[height_start:height_end, width_start:width_end]
        # info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        # info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start

        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"][height_start:height_end, width_start:width_end]
            mask = depth > self.min_depth
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
                # mask = self.eval_mask(mask)
            mask = mask.astype(np.uint8)
            new_gts["gt"] = depth
            new_gts["mask"] = mask
        return image, new_gts, info

    # def eval_mask(self, valid_mask):
    #     border_mask = np.zeros_like(valid_mask)
    #     border_mask[15:465, 20:620] = 1  # prepared center region
    #     return np.logical_and(valid_mask, border_mask)
