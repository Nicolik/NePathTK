import os
import shutil
import logging
import subprocess
import sys
from config_semseg import SelectedConfig
from definitions import OPENSLIDE

os.environ['PATH'] = OPENSLIDE + ";" + os.environ['PATH']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from detection.qupath.defs import PathMESCnn

# Tests
reset_tempdir = True
reset_qupath = True
test_post_processing = True
test_export = True
export_qupath = True
export_json = False
test_qu2json = True
test_json2exp = True

# Config
config = SelectedConfig

cmc_detector_output = os.path.join(config.path_to_export, 'cmc-detector-output')
cmc_aggregator_output = os.path.join(config.path_to_export, 'cmc-aggregator-output')
aag_detector_output = os.path.join(config.path_to_export, 'gaa-detector-output')
aag_aggregator_output = os.path.join(config.path_to_export, 'gaa-aggregator-output')
ifta_detector_output = os.path.join(config.path_to_export, 'ifta-detector-output')
ifta_aggregator_output = os.path.join(config.path_to_export, 'ifta-aggregator-output')

pp_output = os.path.join(config.path_to_export, 'post-processed-masks')
observ_output = os.path.join(config.path_to_export, 'observ-masks')
color_output = os.path.join(config.path_to_export, 'color-masks')

poly2json_output = os.path.join(config.path_to_export, 'poly2json-output')
qu2json_output = os.path.join(config.path_to_export, 'qu2json-output')
json2exp_output = os.path.join(config.path_to_export, 'json2exp-output')

if reset_tempdir:
    for to_del_dir in [pp_output, observ_output, color_output,
                       poly2json_output, qu2json_output, json2exp_output]:
        if os.path.exists(to_del_dir):
            print(f"Removing {to_del_dir}...")
            shutil.rmtree(to_del_dir)
    print("Reset of Temp Dir DONE!")

if reset_qupath:
    if os.path.exists(config.qupath_segm_dir):
        print(f"Removing {config.qupath_segm_dir}...")
        shutil.rmtree(config.qupath_segm_dir)
    print(f"Copying {config.qupath_segm_tocopy_dir} to {config.qupath_segm_dir}...")
    shutil.copytree(config.qupath_segm_tocopy_dir, config.qupath_segm_dir)
    print("Reset of QuPath project DONE!")

wsi_names = os.listdir(aag_detector_output)
wsi_names = [w for w in wsi_names if os.path.isdir(os.path.join(aag_detector_output, w))]
for wsi_name in wsi_names:
    if test_post_processing:
        logging.info(f"{PathMESCnn.POST_PROCESSING_MASK} running on wsi...")
        subprocess.run([sys.executable, PathMESCnn.POST_PROCESSING_MASK,
                        "--wsi", wsi_name,
                        "--cmc-dir", cmc_aggregator_output,
                        "--aag-dir", aag_aggregator_output,
                        # "--ifta-dir", ifta_aggregator_output,
                        "--pp-dir", pp_output,
                        "--observ-dir", observ_output,
                        "--color-dir", color_output
                        ])

    if test_export:
        logging.info(f"{PathMESCnn.EXPORT} running on wsi...")
        params_export = []
        if export_qupath:
            params_export.extend(["--export-qupath", "--qupath-export-dir", config.qupath_segm_dir,])
        if export_json:
            params_export.extend(["--export-json",])
        subprocess.run([sys.executable, PathMESCnn.EXPORT,
                        "--wsi", wsi_name,
                        "--ext", config.ext,
                        "--pp-dir", pp_output,
                        "--wsi-dir", config.wsi_dir,
                        "--undersampling-tissue", str(config.downsample_factor_tissue),
                        "--undersampling-cmc", str(config.downsample_factor_cmc),
                        "--undersampling-aag", str(config.downsample_factor_aag),
                        "--magnification", str(config.magnification),
                        "--base-export-dir", config.path_to_export,
                        *params_export
                        ])

if test_qu2json:
    logging.info(f"{PathMESCnn.QU2JSON} running on wsi...")
    subprocess.run([sys.executable, PathMESCnn.QU2JSON,
                    "--export", config.path_to_export,
                    "--wsi-dir", config.wsi_dir,
                    "--qupath", config.qupath_segm_dir,
                    ])
else:
    logging.info(f"Skipping run of {PathMESCnn.QU2JSON}!")

if test_json2exp:
    logging.info(f"{PathMESCnn.JSON2EXP} running on wsi...")
    subprocess.run([sys.executable, PathMESCnn.JSON2EXP,
                    "--export", config.path_to_export,
                    "--wsi-dir", config.wsi_dir,
                    ])
else:
    logging.info(f"Skipping run of {PathMESCnn.JSON2EXP}!")
