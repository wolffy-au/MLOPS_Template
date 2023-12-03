import os
import sys
current_dir = os.getcwd()
module_path = os.path.join(current_dir, '', 'src')
sys.path.append(module_path)

from x1_data_processing import run_data_processing
from x2_train import run_train
from x3_evaluate import run_evaluate
from x4_deploy import run_deploy
from x5_inference import run_inference
from x6_monitor import run_monitor

def run_main():
    # Run the script using subprocess
    run_data_processing()
    run_train()
    run_evaluate()
    run_deploy()
    run_inference()
    run_monitor()


if __name__ == "__main__":
    run_main()
