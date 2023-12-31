from uc00template.mkdirs import create_directory_structure
from uc00template.x1_data_processing import run_data_processing
from uc00template.x2_train import run_train
from uc00template.x3_evaluate import run_evaluate
from uc00template.x4_deploy import run_deploy
from uc00template.x5_inference import run_inference
from uc00template.x6_monitor import run_monitor
from uc00template.x7_test import run_pytest


def run_main():
    create_directory_structure("uc00template")
    # Run the script using subprocess
    # run_pytest()
    run_data_processing()
    # features = run_data_processing()
    run_train()
    run_evaluate()
    run_deploy()
    run_inference()
    # run_inference(features)
    run_monitor()


if __name__ == "__main__":
    run_main()
