import logging
import os
import javabridge
import bioformats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract tiles from WSI')
    parser.add_argument('-w', '--wsi', type=str, help='path/to/wsi', required=True)
    parser.add_argument('-e', '--export', type=str, help='path/to/export', required=True)
    parser.add_argument('--desired-op', type=float, help='Desired Magnification for performing segmentation', default=10)
    parser.add_argument('--tile-size', type=int, nargs=2, help='Size of a tile (in pixel, at maximum magnification available)', default=(4096, 4096))
    parser.add_argument('--tile-stride', type=int, nargs=2, help='Size of the stride between tiles (in pixel, at maximum magnification available)', default=(2048, 2048))

    args = parser.parse_args()
    desired_op = args.desired_op
    tile_size = args.tile_size
    tile_stride = args.tile_stride
    path_to_wsi = args.wsi
    path_to_tiled = args.export

    print("Starting JVM...")
    javabridge.start_vm(class_path=bioformats.JARS)
    print("JVM started!")

    print("Loading local libraries...")
    from detection.qupath.tile_utils import get_reader_and_tiler
    print("Local libraries loaded!")

    reader, tiler = get_reader_and_tiler(path_to_wsi, path_to_tiled)
    if tiler:
        tiler.tile_image(desired_op, tile_size, tile_stride, check_fg=False)
    del reader
    javabridge.kill_vm()
