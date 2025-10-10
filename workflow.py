import os.path
import subprocess
import sys

from definitions import ROOT_DIR

scripts_dir = os.path.join(ROOT_DIR, 'scripts')

CMC_SEMSEG_SCRIPT = os.path.join(scripts_dir, 'run_wsi_semseg_cmc_bbox.py')
AAG_SEMSEG_SCRIPT = os.path.join(scripts_dir, 'run_wsi_semseg_aag_bbox.py')
PP_SEMSEG_SCRIPT = os.path.join(scripts_dir, 'run_wsi_semseg_pp_exp.py')

subprocess.run([sys.executable, CMC_SEMSEG_SCRIPT])
subprocess.run([sys.executable, AAG_SEMSEG_SCRIPT])
subprocess.run([sys.executable, PP_SEMSEG_SCRIPT])
