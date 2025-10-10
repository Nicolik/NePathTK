import os
import cv2
from scipy import signal
import numpy as np
from PIL import Image
from semseg.qupath.utils import get_coordinates_from_filename_


def aggregate_masks_cmc(masks_directory, new_dimensions, undersampling):
    keys = ["Cortex", "Medulla", "CapsuleOther"]
    return aggregate_masks(masks_directory, new_dimensions, undersampling, keys)


def aggregate_masks_aag(masks_directory, new_dimensions, undersampling):
    keys = ["Glomerulus", "Artery", "Arteriole"]
    return aggregate_masks(masks_directory, new_dimensions, undersampling, keys)


def aggregate_masks_ifta(masks_directory, new_dimensions, undersampling):
    keys = ["IFTACortex"]
    return aggregate_masks(masks_directory, new_dimensions, undersampling, keys)


def get_aggregate_masks(segmentation_type):
    if segmentation_type == "cmc":
        return aggregate_masks_cmc
    elif segmentation_type == "aag":
        return aggregate_masks_aag
    elif segmentation_type == "ifta":
        return aggregate_masks_ifta


def aggregate_masks(masks_directory, new_dimensions, undersampling, keys):
    new_Y, new_X = new_dimensions

    class_masks = {key: np.zeros(new_dimensions, dtype=np.uint8) for key in keys}
    count_masks = {key: np.zeros(new_dimensions, dtype=np.uint8) for key in keys}
    sum_masks = {key: np.zeros(new_dimensions, dtype=np.float32) for key in keys}
    average_masks = {key: np.zeros(new_dimensions, dtype=np.uint8) for key in keys}

    # Aggregate masks
    binary_masks_dir = os.path.join(masks_directory, 'Binary')
    for mask_tile_dir in os.listdir(binary_masks_dir):
        full_tile_path = os.path.join(binary_masks_dir, mask_tile_dir)
        for mask_file in os.listdir(full_tile_path):
            mask_filepath = os.path.join(full_tile_path, mask_file)
            if mask_file.endswith('.png'):
                class_name = os.path.splitext(mask_file)[0]
                tile_mask = np.array(Image.open(mask_filepath))

                coords = get_coordinates_from_filename_(mask_tile_dir)
                x, y, w, h = coords['ROI']
                x = int(x / undersampling)
                y = int(y / undersampling)

                if 'PAD' in coords:
                    pad_top, pad_bottom, pad_left, pad_right = coords['PAD']
                    tile_mask = tile_mask[pad_top:-pad_bottom, pad_left:-pad_right]

                if class_name != 'Background':
                    overall_mask = class_masks[class_name]
                    count_mask = count_masks[class_name]
                    sum_mask = sum_masks[class_name]

                    h_tile, w_tile = tile_mask.shape
                    if y + h_tile > new_Y:
                        h_tile = new_Y - y
                    if x + w_tile > new_X:
                        w_tile = new_X - x
                    overall_mask_tile = overall_mask[y:y + h_tile, x:x + w_tile]
                    tile_mask_tile = tile_mask[:h_tile, :w_tile]
                    ones_tile_mask_tile = (tile_mask_tile > 0).astype(np.uint8)
                    print(f"Writing {y}:{y + h_tile}, {x}:{x + w_tile}")
                    print(f"Tile mask      (tile): {tile_mask_tile.shape}, {tile_mask_tile.min()}-{tile_mask_tile.max()}")
                    print(f"Ones Tile mask (tile): {ones_tile_mask_tile.shape}, {ones_tile_mask_tile.min()}-{ones_tile_mask_tile.max()}")
                    print(f"Overall mask   (tile): {overall_mask_tile.shape}, {overall_mask_tile.min()}-{overall_mask_tile.max()}")
                    overall_mask[y:y + h_tile, x:x + w_tile] = np.maximum(overall_mask_tile, tile_mask_tile)

                    count_mask[y:y + h_tile, x:x + w_tile] += 1
                    # sum_mask[y:y + h_tile, x:x + w_tile] += ones_tile_mask_tile
                    # ht_start, ht_end = h_tile // 4, h_tile * 3 // 4
                    # wt_start, wt_end = w_tile // 4, w_tile * 3 // 4
                    # center_tile_mask_tile = ones_tile_mask_tile[ht_start:ht_end, wt_start:wt_end] * 3
                    # sum_mask[(y + ht_start):(y + ht_end), (x + wt_start):(x + wt_end)] += center_tile_mask_tile
                    weight_2d = get_heaviside_2d(size=tile_mask_tile.shape, side_portion=(1/8, 1/8))
                    sum_mask[y:y + h_tile, x:x + w_tile] += (ones_tile_mask_tile * weight_2d)

    for cls in class_masks:
        if cls != "Background":
            print(f"Sum   (whole): {sum_masks[cls].shape}, {sum_masks[cls].min()}-{sum_masks[cls].max()}")
            print(f"Count (whole): {count_masks[cls].shape}, {count_masks[cls].min()}-{count_masks[cls].max()}")
            # average_masks[cls] = ((sum_masks[cls] / count_masks[cls]) >= 0.5).astype(np.uint8) * 255
            bool_count = (count_masks[cls]).astype(bool)
            average_masks[cls] = (copy_divide(sum_masks[cls], count_masks[cls], bool_count) >= 0.15).astype(np.uint8) * 255
            # average_masks[cls] = (sum_masks[cls] >= 0.5).astype(np.uint8) * 255
            print(f"Average mask (whole): {average_masks[cls].shape}, "
                  f"{average_masks[cls].min()}-{average_masks[cls].max()}, {average_masks[cls].sum()}")

    return class_masks, count_masks, sum_masks, average_masks


def normalize_mask(mask, new_max=255):
    mask = mask.astype(np.float32)
    mask_max = np.max(mask)
    mask_min = np.min(mask)
    if not (mask_max == mask_min == 0):
        mask = (mask-mask_min) / (mask_max-mask_min) * new_max
    return mask.astype(np.uint8)


def copy_divide(a, b, mask):
    out = np.zeros_like(a)
    return np.divide(a, b, out=out, where=mask)


def get_gaussian_2d(size=1024, std=350, filename=None):
    k1d = signal.gaussian(size, std=std).reshape(size, 1)
    kernel = np.outer(k1d, k1d)
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    if filename:
        cv2.imwrite(filename, (kernel * 255).astype(np.uint8))
    return kernel


def get_heaviside_2d(size=(1024, 1024), side_portion=None, step=None, filename=None):
    ysize, xsize = size
    kernel = np.ones((ysize, xsize), dtype=np.float32)
    if step is None:
        if side_portion is None:
            side_portion = (1/4, 1/4)
        yp, xp = side_portion
        ystep = int(ysize * (1 - yp * 2))
        xstep = int(xsize * (1 - xp * 2))
    else:
        ystep, xstep = step
        yp = 1/2 * (1 - ystep/ysize)
        xp = 1/2 * (1 - xstep/xsize)

    sub_kernel = np.ones((ystep, xstep), dtype=np.float32)
    kernel[int(ysize*yp):int(ysize*yp+ystep), int(xsize*xp):int(xsize*xp+xstep)] += sub_kernel
    kernel = kernel / 2
    if filename:
        cv2.imwrite(filename, (kernel * 255).astype(np.uint8))
    return kernel
