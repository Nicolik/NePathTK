import logging
import os
import javabridge
import bioformats

from detection.io.config import MULTI_IMAGE_EXTENSIONS
from detection.qupath.utils import bbox2xywh, get_wsi_json_info, get_wsi_multi_json_info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract tiles from WSI')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi', required=True)
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('-t', '--tissue-tiler-dir', type=str, help='path/to/tissue/tiler/dir', required=True)
    parser.add_argument('--desired-op', type=float, help='Desired Magnification for performing segmentation', default=10)
    parser.add_argument('--tile-size', type=int, nargs=2, help='Size of a tile (in pixel, at maximum magnification available)', default=(4096, 4096))
    parser.add_argument('--tile-stride', type=int, nargs=2, help='Size of the stride between tiles (in pixel, at maximum magnification available)', default=(2048, 2048))

    args = parser.parse_args()
    desired_op = args.desired_op
    tile_size = args.tile_size
    tile_stride = args.tile_stride
    path_to_wsi = args.wsi
    path_to_tiled = args.export
    tissue_tiler_dir = args.tissue_tiler_dir

    print("Starting JVM...")
    javabridge.start_vm(class_path=bioformats.JARS)
    print("JVM started!")

    print("Loading local libraries...")
    from detection.qupath.tile_utils import get_reader_and_tiler
    print("Local libraries loaded!")

    reader, tiler = get_reader_and_tiler(path_to_wsi, path_to_tiled)
    if tiler:
        if path_to_wsi.endswith(MULTI_IMAGE_EXTENSIONS):
            list_dict_metadata = get_wsi_multi_json_info(path_to_wsi, tissue_tiler_dir)
            for i, dict_metadata in enumerate(list_dict_metadata):
                if "bounding_boxes" in dict_metadata:
                    bboxes = dict_metadata["bounding_boxes"]
                    for bbox in bboxes:
                        roi = bbox2xywh(bbox)
                        print(f"WSI: {path_to_wsi}, tiling {roi}")
                        tiler.tile_roi(roi, desired_op, tile_size, tile_stride, i=i, check_fg=False, export_json=False)
        else:
            dict_metadata = get_wsi_json_info(path_to_wsi, tissue_tiler_dir)
            if "bounding_boxes" in dict_metadata:
                bboxes = dict_metadata["bounding_boxes"]
                for bbox in bboxes:
                    roi = bbox2xywh(bbox)
                    print(f"WSI: {path_to_wsi}, tiling {roi}")
                    tiler.tile_roi(roi, desired_op, tile_size, tile_stride, check_fg=False, export_json=False)
    del reader
    javabridge.kill_vm()
