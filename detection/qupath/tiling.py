import json
import os
import cv2
import shutil
from detection.qupath.utils import tile_region, is_foreground, is_black, dir_name_from_wsi, is_not_too_small


class BaseTiler(object):
    def __init__(self, reader, out_dir, tile_ext='jpeg', suffix=False, force_reprocess=False):
        self.reader = reader
        to_process = True
        if not suffix:
            out_dir = os.path.join(out_dir, dir_name_from_wsi(reader.name))
            if os.path.exists(out_dir):
                if force_reprocess:
                    shutil.rmtree(out_dir)
                else:
                    to_process = False
            os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.tile_ext = tile_ext
        self.suffix = suffix
        self.to_process = to_process

    def tile_roi(self, roi, desired_op, tile_size, tile_stride):
        pass

    def tile_image(self, desired_op, tile_size, tile_stride):
        pass


class WholeTilerOpenslide(BaseTiler):
    def __init__(self, reader, out_dir, tile_ext='jpeg', suffix=False):
        super().__init__(reader, out_dir, tile_ext=tile_ext, suffix=suffix)

    def tile_roi(self, roi, desired_op, tile_size, tile_stride, check_fg=True, export_json=True, pad_small=True):
        if not self.to_process:
            print(f"Directory '{self.out_dir}' already exists, not reprocessing!")
            return

        WW, HH = tile_size
        tile_coords = tile_region(roi, tile_size, tile_stride, small_tiles=False)
        for tile_coord in tile_coords:
            x, y, ww, hh = tile_coord
            if ww > 64 and hh > 64:
                # tile_image = self.reader.read_resolution(x, y, ww, hh, desired_op, do_rescale=True, read_bgr=True)
                tile_image = self.reader.read_resolution_or_rescale(x, y, ww, hh, desired_op, read_bgr=True)
                tile_name = f"OP_{desired_op}__ROI_{x}_{y}_{ww}_{hh}.{self.tile_ext}"
                if is_not_too_small(tile_image):
                    if not is_black(tile_image):
                        if not check_fg or is_foreground(tile_image):
                            tile_path = os.path.join(self.out_dir, tile_name)
                            print(f"Writing to {tile_path}")
                            if pad_small:
                                if ww < WW or hh < HH:
                                    bottom = HH - hh
                                    right = WW - ww

                                    real_hh, real_ww, _ = tile_image.shape
                                    scale_h = hh / real_hh
                                    scale_w = ww / real_ww

                                    top_scaled = 0
                                    bottom_scaled = int(bottom/scale_h)
                                    left_scaled = 0
                                    right_scaled = int(right/scale_w)

                                    print(f"Before padding: {tile_image.shape}")
                                    tile_image = cv2.copyMakeBorder(tile_image, top_scaled, bottom_scaled,
                                                                    left_scaled, right_scaled,
                                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
                                    print(f"After padding: {tile_image.shape}")
                            cv2.imwrite(tile_path, tile_image)
                        else:
                            print(f"[tile_roi][warning] {tile_name} is background!")
                    else:
                        print(f"[tile_roi][warning] {tile_name} is black!")

        dims = self.reader.get_dimensions()
        json_out = f"{self.out_dir}.json"
        print(f"[tile_roi] The tiler has generated {len(os.listdir(self.out_dir))} output tiles")
        print(f"[tile_roi] Writing metadata to {json_out}")
        with open(json_out, "w") as fp:
            json.dump({"X": dims[0], "Y": dims[1]}, fp)

    def tile_image(self, desired_op, tile_size, tile_stride, check_fg=True, export_json=True):
        roi = (0, 0, *self.reader.dimensions)
        self.tile_roi(roi, desired_op, tile_size, tile_stride, check_fg=check_fg, export_json=export_json)


class WholeTilerBioformats(BaseTiler):
    def __init__(self, reader, out_dir, tile_ext='jpeg', suffix=False):
        super().__init__(reader, out_dir, tile_ext=tile_ext, suffix=suffix)

    def tile_roi(self, roi, desired_op, tile_size, tile_stride, i=0, check_fg=True, export_json=True):
        if not self.to_process:
            print(f"Directory '{self.out_dir}' already exists, not reprocessing!")
            return

        # for i, dimensions in enumerate(self.reader.dimensions):
        if self.suffix:
            scene_dir = f"{dir_name_from_wsi(self.reader.name)}_{i}"
            scene_out_dir = os.path.join(self.out_dir, scene_dir)
            os.makedirs(scene_out_dir, exist_ok=True)
        else:
            scene_out_dir = self.out_dir
        s = self.reader.indexes[i]
        print(f"[tile_roi] i: {i}, s: {s}, roi: {roi}")
        tile_coords = tile_region(roi, tile_size, tile_stride)
        for tile_coord in tile_coords:
            x, y, ww, hh = tile_coord
            if ww > 64 and hh > 64:
                tile_image = self.reader.read_resolution_or_rescale(s, x, y, ww, hh, desired_op, read_bgr=True)
                tile_name = f"{self.reader.name}_{s}__OP_{desired_op}__ROI_{x}_{y}_{ww}_{hh}.{self.tile_ext}"
                not_none = tile_image is not None
                if not_none:
                    not_too_small = is_not_too_small(tile_image)
                    not_black = not is_black(tile_image)
                    yes_fg = not check_fg or is_foreground(tile_image)
                else:
                    not_too_small = not_black = yes_fg = False
                if not_too_small and not_black and yes_fg:
                    tile_path = os.path.join(scene_out_dir, tile_name)
                    print(f"Writing to {tile_path}")
                    cv2.imwrite(tile_path, tile_image)
                print(f"[tile_image] Checks. Name: {tile_name}\n"
                      f"Not None       -> {not_none}\n"
                      f"Not too small  -> {not_too_small}\n"
                      f"Not black      -> {not_black}\n"
                      f"Yes foreground -> {yes_fg}")

        dims = self.reader.get_dimensions(i)
        json_out = f"{scene_out_dir}.json"
        print(f"[tile_roi] The tiler has generated {len(os.listdir(scene_out_dir))} output tiles")
        print(f"[tile_roi] Writing metadata to {json_out}")
        with open(json_out, "w") as fp:
            json.dump({"X": dims[0], "Y": dims[1]}, fp)

    def tile_image(self, desired_op, tile_size, tile_stride, check_fg=True, export_json=True):
        for i, dimensions in enumerate(self.reader.dimensions):
            roix, roiy = dimensions
            roi = (0, 0, roix, roiy)
            self.tile_roi(roi, desired_op, tile_size, tile_stride, i=i, check_fg=check_fg, export_json=export_json)
