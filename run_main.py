from scripts.x1_data_processing import run_data_processing
from scripts.x2_train import run_train
from scripts.x3_evaluate import run_evaluate
from scripts.x4_deploy import run_deploy
from scripts.x5_inference import run_inference
from scripts.x6_monitor import run_monitor
from scripts.x7_test import run_pytest

def run_main():
    # Run the script using subprocess
    run_pytest()
    run_data_processing()
    run_train()
    run_evaluate()
    run_deploy()
    run_inference()
    run_monitor()

if __name__ == "__main__":
    run_main()
