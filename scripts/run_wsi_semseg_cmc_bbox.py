import os
import logging
import subprocess
import sys
from config_semseg import SelectedConfig, SemSegConfigCMC
from definitions import OPENSLIDE

os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from detection.qupath.defs import PathMESCnn

# Tests
test_tile = False
test_semseg_cmc = True
use_mm = True
test_aggr_wsi_cmc_bin_bbox = True

# Config
config = SelectedConfig
semseg = SemSegConfigCMC

cmc_output_tiler = os.path.join(config.path_to_export, 'cmc-tiler-output-bbox')
cmc_detector_output = os.path.join(config.path_to_export, 'cmc-detector-output')
cmc_aggregator_output = os.path.join(config.path_to_export, 'cmc-aggregator-output')

os.makedirs(cmc_aggregator_output, exist_ok=True)
os.makedirs(cmc_detector_output, exist_ok=True)

if test_tile:
    OVERLAP = 50
    TILE_SIZE = int(1024 * config.downsample_factor_cmc)
    TILE_STRIDE = int(TILE_SIZE - (TILE_SIZE * OVERLAP / 100))
    for wsi in config.wsis:
        print(f"{PathMESCnn.TILE} running on {wsi}...")
        subprocess.run([sys.executable, PathMESCnn.TILE,
                        "--wsi", wsi,
                        "--export", cmc_output_tiler,
                        "--desired-op", str(config.desired_op_cmc),
                        "--tile-size", str(TILE_SIZE), str(TILE_SIZE),
                        "--tile-stride", str(TILE_STRIDE), str(TILE_STRIDE),
                        ])
else:
    logging.info(f"Skipping run of {PathMESCnn.TILE}!")

if test_semseg_cmc:
    tiles_dir = os.listdir(cmc_output_tiler)
    tiles_dir = [os.path.join(cmc_output_tiler, t) for t in tiles_dir
                 if os.path.isdir(os.path.join(cmc_output_tiler, t))]
    segments_dir = [t.replace('cmc-tiler-output-bbox', 'cmc-detector-output') for t in tiles_dir]

    for tile_dir, segment_dir in zip(tiles_dir, segments_dir):
        print(f"{semseg.semseg_cmc} running on tile-dir: {tile_dir}, segment-dir: {segment_dir}")
        cmd = [sys.executable, semseg.semseg_cmc,
               "--tile", tile_dir,
               "--output", segment_dir,
               "--batch-size", str(semseg.batch_size),
               "--n-cpu", str(semseg.n_cpu),
               "--arch", semseg.arch,
               "--enc-name", semseg.enc_name,
               "--microscopy-type", semseg.microscopy_type,
               ]
        if semseg.use_tta:
            cmd += ["--use-tta"]
        subprocess.run(cmd)
else:
    logging.info(f"Skipping run of {semseg.semseg_cmc}!")

if test_aggr_wsi_cmc_bin_bbox:
    wsi_names = os.listdir(cmc_detector_output)
    wsi_names = [w for w in wsi_names if os.path.isdir(os.path.join(cmc_detector_output, w))]
    for wsi_name in wsi_names:
        json_file = wsi_name + '.json'
        json_metadata = os.path.join(cmc_output_tiler, json_file)
        logging.info(f"{PathMESCnn.AGGREGATE_WSI_BIN_BBOX} running on wsi...")
        subprocess.run([sys.executable, PathMESCnn.AGGREGATE_WSI_BIN_BBOX,
                        "--wsi", wsi_name,
                        "--ext", config.ext,
                        "--masks", cmc_detector_output,
                        "--wsi-masks", cmc_aggregator_output,
                        "--segmentation-type", "cmc",
                        "--undersampling", str(config.downsample_factor_cmc),
                        "--json-metadata", json_metadata,
                        ])
else:
    logging.info(f"Skipping run of {PathMESCnn.AGGREGATE_WSI_BIN_BBOX}!")
