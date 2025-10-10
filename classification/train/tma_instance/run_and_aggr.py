# python .\classification\train\tma_instance\run_and_aggr.py
import sys
import subprocess

run_path = "classification/train/tma_instance/run.py"
aggr_path = "classification/train/tma_instance/cval_aggr.py"

subprocess.run([sys.executable, run_path])
subprocess.run([sys.executable, aggr_path])
