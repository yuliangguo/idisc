"""
Author: Yuliang Guo
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import numpy as np
import cv2
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
        tgt_f=7.215377e02,
        undistort_f=0,
        fwd_sz=(700, 700),
        erp=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        self.is_dense = is_dense
        self.tgt_f = tgt_f
        self.undistort_f = undistort_f
        self.fwd_sz = fwd_sz
        self.erp = erp

        if undistort_f >0:
            self.split_file = self.split_file[:-4] + f"_undistort_f{self.undistort_f}.txt"
        if erp:
            self.split_file = self.split_file[:-4] + "_erp.txt"

        # if resize_im:
        #     # the dataset will be resized to match the focal length of the kitti dataset
        #     if undistort_f >0:
        #         self.height = int(1400 * tgt_f / undistort_f)
        #         self.width = int(1400 * tgt_f / undistort_f)
        #     else:
        #         self.height = int(1400 * tgt_f / self.CAM_INTRINSIC['02'][1, 1])
        #         self.width = int(1400 * tgt_f / self.CAM_INTRINSIC['02'][1, 1])
        # else:
        self.height = fwd_sz[0]
        self.width = fwd_sz[1]            

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
                    
                # setup original intrinsics
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                if 'image_02' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['02'][:, :3].clone()
                elif 'image_03' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['03'][:, :3].clone()
                
                if self.undistort_f > 0:
                    img_info["camera_intrinsics"] = torch.tensor(
                        [
                            [self.undistort_f, 0.000000e00, 700.],
                            [0.000000e00, self.undistort_f, 700.],
                            [0.000000e00, 0.000000e00, 1.000000e00],
                        ])
                elif self.erp:
                    img_info["camera_intrinsics"] = torch.tensor(
                        [
                            [1 / np.tan(np.pi/1400), 0.000000e00, 700.],
                            [0.000000e00, 1 / np.tan(np.pi/1400), 700.],
                            [0.000000e00, 0.000000e00, 1.000000e00],
                        ])
                
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

        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        ori_h, ori_w, _ = image.shape
        image, depth, pad, pred_scale_factor = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], [ori_h, ori_w], 1.0)
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        if not self.test_mode:
            depth /= info['pred_scale_factor']

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


def resize_for_input(image, depth, output_shape, intrinsic, canonical_shape, to_canonical_ratio):
    """
    Resize the input.
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    
    If the image is the original image, just set to_canonical_ratio=1, canonical_shape as the original image shape.
    """
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)

    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_NEAREST)
    # padding
    image = cv2.copyMakeBorder(
        image, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=padding)
    depth = cv2.copyMakeBorder(
        depth, 
        pad_h_half, 
        pad_h - pad_h_half, 
        pad_w_half, 
        pad_w - pad_w_half, 
        cv2.BORDER_CONSTANT, 
        value=0)
    
    # Resize, adjust principle point
    intrinsic[0, 2] = intrinsic[0, 2] * to_scale_ratio
    intrinsic[1, 2] = intrinsic[1, 2] * to_scale_ratio

    pad=[pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    # label_scale_factor=1/to_scale_ratio
    return image, depth, pad, to_scale_ratio