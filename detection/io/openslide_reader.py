import os
import numpy as np
import cv2
import openslide

from detection.io.config import check_desired_op
from detection.qupath.utils import find_nearest, PIL2cv2, find_nearest_greater


class OpenslideReader(object):
    def __init__(self, path_to_wsi):
        self.path_to_wsi = path_to_wsi
        self.name = os.path.basename(path_to_wsi)
        self.image_os = openslide.open_slide(path_to_wsi)
        try:
            self.objective_power = int(self.image_os.properties['openslide.objective-power'])
        except KeyError as E:
            print(f"KeyError: {E}, defaulting OP to 40!")
            self.objective_power = 40
        self.dimensions = self.image_os.dimensions
        self.objective_powers = []
        self.pixels_left = 0
        self.pixels_top = 0
        self._rescale_factor = None
        self._init_objective_powers()
        self._init_pixels_offset()

    def get_dimensions(self):
        return self.dimensions

    def _init_pixels_offset(self):
        prop_left = None
        prop_top = None
        for prop in self.image_os.properties:
            if 'COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__LEFT' in prop:
                prop_left = prop
            elif 'COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS__TOP' in prop:
                prop_top = prop
        self.pixels_left = int(self.image_os.properties[prop_left]) if prop_left else 0
        self.pixels_top = int(self.image_os.properties[prop_top]) if prop_top else 0
        print(f"[_init_pixels_offset] displacement: (T) {self.pixels_top}, (L) {self.pixels_left}")

    def _init_objective_powers(self):
        level_dims = self.image_os.level_dimensions
        obj_powers = []
        for level_dim in level_dims:
            obj_powers.append(self.objective_power * level_dim[0] / self.dimensions[0])
        self.objective_powers = np.array(obj_powers)

    def read_region(self, location, level, size, read_bgr=False):
        x, y = location
        xsize, ysize = size
        print(f"[read_region] reading ({x + self.pixels_left}, {y + self.pixels_top}), {level}, ({xsize}, {ysize})")
        crop = self.image_os.read_region((x + self.pixels_left, y + self.pixels_top), level, (xsize, ysize))
        crop = PIL2cv2(crop)
        if read_bgr:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        return crop

    def read_resolution_or_rescale(self, x, y, w, h, desired_op, read_bgr=False):
        desired_tile = self.read_resolution(x, y, w, h, desired_op, read_bgr=read_bgr)
        if desired_tile is None:
            closest_level, closest_op = find_nearest_greater(self.objective_powers, desired_op)
            tile_closest = self.read_resolution(x, y, w, h, closest_op, read_bgr=read_bgr)
            rescale_factor = closest_op / desired_op
            print(f"[read_resolution_or_rescale] Rescale factor: {rescale_factor:.2f}")
            # w = int(w / round(rescale_factor, 1))
            # h = int(h / round(rescale_factor, 1))
            # tile_closest = cv2.resize(tile_closest, (w, h), interpolation=cv2.INTER_LINEAR)
            tile_closest = cv2.resize(tile_closest, None, fx=1 / rescale_factor, fy=1 / rescale_factor,
                                      interpolation=cv2.INTER_LINEAR)
            return tile_closest
        return desired_tile

    def read_resolution(self, x, y, w, h, desired_op, do_rescale=True, read_bgr=False):
        closest_level, closest_op = find_nearest_greater(self.objective_powers, desired_op)
        if not check_desired_op(closest_op, desired_op):
            print(f"Mismatch between closest OP and desired OP, {closest_op} vs {desired_op}")
            return None
        rescale_factor = self.objective_power / closest_op
        self._rescale_factor = rescale_factor
        print(f"[read_resolution] Rescale Factor: {rescale_factor}")
        if do_rescale:
            print(f"[read_resolution] Before Rescaling: [{x}, {y}, {w}, {h}]")
            x, y, w, h = int(x), int(y), int(w/round(rescale_factor, 1)), int(h/round(rescale_factor, 1))
            print(f"[read_resolution] After  Rescaling: [{x}, {y}, {w}, {h}]")
        return self.read_region((x, y), closest_level, (w, h), read_bgr=read_bgr)
