"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class KITTIERPDataset(BaseDataset):
    CAM_INTRINSIC = {
        "ALL": torch.tensor(
            [
                [1 / np.tan(np.pi/1400), 0.000000e00, 700.],
                [0.000000e00, 1 / np.tan(np.pi/1400), 700.],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti_eigen_test.txt"
    train_split = "kitti_eigen_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.is_dense = is_dense
        self.height = 256
        self.width = 704

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
                    if depth_map == "None" or not os.path.exists(
                        os.path.join(self.base_path, depth_map)
                    ):
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
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
            Image.open(
                os.path.join(
                    # self.base_path, 
                    self.dataset[idx]["image_filename"])
            )
        ).astype(np.uint8)
        depth = None
        if not self.benchmark:
            depth = (
                np.asarray(
                    Image.open(
                        os.path.join(
                            # self.base_path,
                            self.dataset[idx]["annotation_filename_depth"],
                        )
                    )
                ).astype(np.float32)
                / self.depth_scale
            )
        info = self.dataset[idx].copy()
        info["camera_intrinsics"] = self.CAM_INTRINSIC["ALL"].clone()
        image, gts, info = self.transform(image=image, gts={"depth": depth}, info=info)
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"]}
        
    # def get_pointcloud_mask(self, shape):
    #     if self.crop is None:
    #         return np.ones(shape)
    #     mask_height, mask_width = shape
    #     mask = np.zeros(shape)
    #     if "garg" in self.crop:
    #         mask[
    #             int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
    #             int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #         ] = 1
    #     elif "eigen" in self.crop:
    #         mask[
    #             int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
    #             int(0.0359477 * mask_width) : int(0.96405229 * mask_width),
    #         ] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = int(image.shape[0] - self.height), int(
            (image.shape[1] - self.width) / 2
        )
        height_end, width_end = height_start + self.height, width_start + self.width
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            if depth is not None:
                height_start, width_start = int(depth.shape[0] - self.height), int(
                    (depth.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                depth = depth[height_start:height_end, width_start:width_end]
                mask = depth > self.min_depth
                if self.test_mode:
                    mask = np.logical_and(mask, depth < self.max_depth)
                    mask = self.eval_mask(mask)
                mask = mask.astype(np.uint8)
                new_gts["gt"] = depth
                new_gts["mask"] = mask

        return image, new_gts, info

    # def eval_mask(self, valid_mask):
    #     """Do grag_crop or eigen_crop for testing"""
    #     if self.test_mode:
    #         if self.crop is not None:
    #             mask_height, mask_width = valid_mask.shape[-2:]
    #             eval_mask = np.zeros_like(valid_mask)
    #             if "garg" in self.crop:
    #                 eval_mask[
    #                     int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
    #                     int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #                 ] = 1
    #             elif "eigen" in self.crop:
    #                 eval_mask[
    #                     int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
    #                     int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #                 ] = 1
    #         valid_mask = np.logical_and(valid_mask, eval_mask)
    #     return valid_mask
