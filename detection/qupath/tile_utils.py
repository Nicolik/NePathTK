import os
from definitions import OPENSLIDE
os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']
import openslide

from detection.io.bioformats_reader import BioformatsReader
from detection.io.config import OPENSLIDE_EXTENSIONS, BIOFORMATS_EXTENSIONS, MULTI_IMAGE_EXTENSIONS
from detection.io.openslide_reader import OpenslideReader
from detection.qupath.tiling import WholeTilerBioformats, WholeTilerOpenslide


def get_reader_and_tiler(path_to_wsi, path_to_tiled):
    if path_to_wsi.endswith(BIOFORMATS_EXTENSIONS):
        reader = BioformatsReader(path_to_wsi)
        suffix = path_to_wsi.endswith(MULTI_IMAGE_EXTENSIONS)
        tiler = WholeTilerBioformats(reader, path_to_tiled, suffix=suffix)
    elif path_to_wsi.endswith(OPENSLIDE_EXTENSIONS):
        try:
            reader = OpenslideReader(path_to_wsi)
            tiler = WholeTilerOpenslide(reader, path_to_tiled)
        except openslide.lowlevel.OpenSlideError as E:
            print(f"OpenSlideError: {E}")
            reader = tiler = None
    else:
        print(f"[get_reader_and_tiler] Warning! Extension of {path_to_wsi} not supported!")
        reader = tiler = None
    return reader, tiler
