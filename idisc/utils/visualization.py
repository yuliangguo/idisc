from typing import Any, Dict, List, Tuple

import matplotlib.cm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, cv2
import torch


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    if value.ndim > 2:
        return value
    invalid_mask = value == -1

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 255
    img = value[..., :3]
    return img


def image_grid(imgs: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    if not len(imgs):
        return None

    assert len(imgs) == rows * cols
    h, w = imgs[0].shape[:2]
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(
            Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.ANTIALIAS),
            box=(i % cols * w, i // cols * h),
        )

    return np.array(grid)


def get_pointcloud_from_rgbd(
    image: np.array,
    depth: np.array,
    mask: np.ndarray,
    intrinsic_matrix: np.array,
    extrinsic_matrix: np.array = None,
):
    depth = np.array(depth).squeeze()
    mask = np.array(mask).squeeze()
    # Mask the depth array
    masked_depth = np.ma.masked_where(mask == False, depth)
    # masked_depth = np.ma.masked_greater(masked_depth, 8000)
    # Create idx array
    idxs = np.indices(masked_depth.shape)
    u_idxs = idxs[1]
    v_idxs = idxs[0]
    # Get only non-masked depth and idxs
    z = masked_depth[~masked_depth.mask]
    compressed_u_idxs = u_idxs[~masked_depth.mask]
    compressed_v_idxs = v_idxs[~masked_depth.mask]
    image = np.stack(
        [image[..., i][~masked_depth.mask] for i in range(image.shape[-1])], axis=-1
    )

    # Calculate local position of each point
    # Apply vectorized math to depth using compressed arrays
    cx = intrinsic_matrix[0, 2]
    fx = intrinsic_matrix[0, 0]
    x = (compressed_u_idxs - cx) * z / fx
    cy = intrinsic_matrix[1, 2]
    fy = intrinsic_matrix[1, 1]
    # Flip y as we want +y pointing up not down
    y = -((compressed_v_idxs - cy) * z / fy)

    # # Apply camera_matrix to pointcloud as to get the pointcloud in world coords
    if extrinsic_matrix is not None:
        # Calculate camera pose from extrinsic matrix
        camera_matrix = np.linalg.inv(extrinsic_matrix)
        # Create homogenous array of vectors by adding 4th entry of 1
        # At the same time flip z as for eye space the camera is looking down the -z axis
        w = np.ones(z.shape)
        x_y_z_eye_hom = np.vstack((x, y, -z, w))
        # Transform the points from eye space to world space
        x_y_z_world = np.dot(camera_matrix, x_y_z_eye_hom)[:3]
        return x_y_z_world.T
    else:
        x_y_z_local = np.stack((x, y, z), axis=-1)
    return np.concatenate([x_y_z_local, image], axis=-1)


def save_file_ply(xyz, rgb, pc_file):
    if rgb.max() < 1.001:
        rgb = rgb * 255.0
    rgb = rgb.astype(np.uint8)
    # print(rgb)
    with open(pc_file, "w") as f:
        # headers
        f.writelines(
            [
                "ply\n" "format ascii 1.0\n",
                "element vertex {}\n".format(xyz.shape[0]),
                "property float x\n",
                "property float y\n",
                "property float z\n",
                "property uchar red\n",
                "property uchar green\n",
                "property uchar blue\n",
                "end_header\n",
            ]
        )

        for i in range(xyz.shape[0]):
            str_v = "{:10.6f} {:10.6f} {:10.6f} {:d} {:d} {:d}\n".format(
                xyz[i, 0], xyz[i, 1], xyz[i, 2], rgb[i, 0], rgb[i, 1], rgb[i, 2]
            )
            f.write(str_v)


# visualization code copied from Metric3D
def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap


def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()

    pred[pred<0] = 0
    target[target<0] = 0
    max_scale = max(pred.max(), target.max())
    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred)
    target_color = gray_to_colormap(target)
    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, target_scale, pred_color, target_color


def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor,
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color = get_data_for_log(pred, target, rgb)
    rgb = rgb.transpose((1, 2, 0))
    cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)