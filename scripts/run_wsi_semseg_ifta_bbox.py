import os
import logging
import subprocess
import sys
from config_semseg import SelectedConfig, SemSegConfigIFTA
from definitions import OPENSLIDE

os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from detection.qupath.defs import PathMESCnn


# Tests
test_tile_bbox = True
test_semseg_ifta = True
use_mm = True
test_aggr_wsi_ifta_bin_bbox = True
test_qu2json = False
test_json2exp = False

# Config
config = SelectedConfig
semseg = SemSegConfigIFTA

cmc_output_tiler = os.path.join(config.path_to_export, 'cmc-tiler-output-bbox')
aag_output_tiler = os.path.join(config.path_to_export, 'gaa-tiler-output-bbox')
ifta_detector_output = os.path.join(config.path_to_export, 'ifta-detector-output')
ifta_aggregator_output = os.path.join(config.path_to_export, 'ifta-aggregator-output')

os.makedirs(ifta_aggregator_output, exist_ok=True)
os.makedirs(ifta_detector_output, exist_ok=True)

if test_tile_bbox:
    OVERLAP = 50
    TILE_SIZE = int(1024 * config.downsample_factor_aag)
    TILE_STRIDE = int(TILE_SIZE - (TILE_SIZE * OVERLAP / 100))
    for wsi in config.wsis:
        logging.info(f"{PathMESCnn.TILE_BBOX} running on {wsi}...")
        subprocess.run([sys.executable, PathMESCnn.TILE_BBOX,
                        "--wsi", wsi,
                        "--export", aag_output_tiler,
                        "--tissue-tiler-dir", cmc_output_tiler,
                        "--desired-op", str(config.desired_op_aag),
                        "--tile-size", str(TILE_SIZE), str(TILE_SIZE),
                        "--tile-stride", str(TILE_STRIDE), str(TILE_STRIDE),
                        ])
else:
    logging.info(f"Skipping run of {PathMESCnn.TILE_BBOX}!")

if test_semseg_ifta:
    tiles_dir = os.listdir(aag_output_tiler)
    tiles_dir = [os.path.join(aag_output_tiler, t) for t in tiles_dir
                 if os.path.isdir(os.path.join(aag_output_tiler, t))]
    segments_dir = [t.replace('gaa-tiler-output-bbox', 'ifta-detector-output') for t in tiles_dir]

    for tile_dir, segment_dir in zip(tiles_dir, segments_dir):
        print(f"{semseg.semseg_aag} running on tile-dir: {tile_dir}, segment-dir: {segment_dir}")
        cmd = [sys.executable, semseg.semseg_ifta,
               "--tile", tile_dir,
               "--output", segment_dir,
               "--batch-size", str(semseg.batch_size),
               "--n-cpu", str(semseg.n_cpu),
               "--arch", semseg.arch,
               "--enc-name", semseg.enc_name,
               "--microscopy-type", semseg.microscopy_type,
               ]
        if semseg.use_tta:
            cmd +=  ["--use-tta"]
        subprocess.run(cmd)
else:
    logging.info(f"Skipping run of {semseg.semseg_ifta}!")

if test_aggr_wsi_ifta_bin_bbox:
    wsi_names = os.listdir(ifta_detector_output)
    wsi_names = [w for w in wsi_names if os.path.isdir(os.path.join(ifta_detector_output, w))]
    for wsi_name in wsi_names:
        json_file = wsi_name + '.json'
        json_metadata = os.path.join(cmc_output_tiler, json_file)
        logging.info(f"{PathMESCnn.AGGREGATE_WSI_BIN_BBOX} running on wsi...")
        subprocess.run([sys.executable, PathMESCnn.AGGREGATE_WSI_BIN_BBOX,
                        "--wsi", wsi_name,
                        "--ext", config.ext,
                        "--masks", ifta_detector_output,
                        "--wsi-masks", ifta_aggregator_output,
                        "--segmentation-type", "ifta",
                        "--undersampling", str(config.downsample_factor_aag),
                        "--json-metadata", json_metadata,
                        ])
else:
    logging.info(f"Skipping run of {PathMESCnn.AGGREGATE_WSI_BIN_BBOX}!")
