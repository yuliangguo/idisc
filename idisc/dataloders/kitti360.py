"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset


class KITTI360Dataset(BaseDataset):
    CAM_INTRINSIC = {
        "02": torch.tensor(
            [
                [1.33632e+03, 0.000000e00, 7.16943e+02],
                [0.000000e00, 1.33578e+03, 7.05764e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        ),
        "03": torch.tensor(
            [
                [1.48543e+03, 0.000000e00, 6.98883e+02],
                [0.000000e00, 1.48494e+03, 6.98145e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    
    camera_params_02 = {
        "model_type": "MEI",
        "camera_name": "image_02",
        "image_width": 1400,
        "image_height": 1400,
        "mirror_parameters": {
            "xi": 2.2134047507854890e+00
        },
        "distortion_parameters": {
            "k1": 1.6798235660113681e-02,
            "k2": 1.6548773243373522e+00,
            "p1": 4.2223943394772046e-04,
            "p2": 4.2462134260997584e-04
        },
        "projection_parameters": {
            "gamma1": 1.3363220825849971e+03,
            "gamma2": 1.3357883350012958e+03,
            "u0": 7.1694323510126321e+02,
            "v0": 7.0576498308221585e+02
        }
    }
    camera_params_03 = {
        "model_type": "MEI",
        "camera_name": "image_03",
        "image_width": 1400,
        "image_height": 1400,
        "mirror_parameters": {
            "xi": 2.5535139132482758e+00
        },
        "distortion_parameters": {
            "k1": 4.9370396274089505e-02,
            "k2": 4.5068455478645308e+00,
            "p1": 1.3477698472982495e-03,
            "p2": -7.0340482615055284e-04
        },
        "projection_parameters": {
            "gamma1": 1.4854388981875156e+03,
            "gamma2": 1.4849477411748708e+03,
            "u0": 6.9888316784030962e+02,
            "v0": 6.9814541887723055e+02
        }
    }
    
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti360_val_fisheye.txt"
    train_split = "kitti360_train_fisheye.txt"

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
        tgt_fy = 7.215377e02,
        undistort_f = 350,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        self.is_dense = is_dense
        self.tgt_fy = tgt_fy
        self.undistort_f = undistort_f

        # the dataset will be resized to match the focal length of the kitti dataset
        if undistort_f is not None:
            self.height = int(1400 * tgt_fy / undistort_f)
            self.width = int(1400 * tgt_fy / undistort_f)
            self.split_file = self.split_file[:-4] + f"_undistort_f{self.undistort_f}.txt"
        else:
            self.height = int(1400 * tgt_fy / self.CAM_INTRINSIC['02'][1, 1])
            self.width = int(1400 * tgt_fy / self.CAM_INTRINSIC['02'][1, 1])

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
                if 'image_02' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['02'][:, :3]
                elif 'image_03' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['03'][:, :3]
                
                if self.undistort_f is not None:
                    img_info["camera_intrinsics"] = torch.tensor(
                        [
                            [self.undistort_f, 0.000000e00, 700.],
                            [0.000000e00, self.undistort_f, 700.],
                            [0.000000e00, 0.000000e00, 1.000000e00],
                        ])
                    
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
            ).resize((self.width, self.height))
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
                    ).resize((self.width, self.height), Image.NEAREST)
                ).astype(np.float32)
                / self.depth_scale
            )
        info = self.dataset[idx].copy()

        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        scaler = self.tgt_fy / info["camera_intrinsics"][1, 1]
        info["camera_intrinsics"][0, 0] *= scaler
        info["camera_intrinsics"][1, 1] *= scaler
        info["camera_intrinsics"][0, 2] *= scaler
        info["camera_intrinsics"][1, 2] *= scaler
        image, gts, info = self.transform(image=image, gts={"depth": depth}, info=info)
        return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "info": info}

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
