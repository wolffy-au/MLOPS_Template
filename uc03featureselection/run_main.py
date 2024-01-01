from uc03featureselection.mkdirs import create_directory_structure
from uc03featureselection.x1_data_processing import run_data_processing
from uc03featureselection.x2_train import run_train
from uc03featureselection.x3_evaluate import run_evaluate
from uc03featureselection.x4_deploy import run_deploy
from uc03featureselection.x5_inference import run_inference
from uc03featureselection.x6_monitor import run_monitor
from uc03featureselection.x7_test import run_pytest


def run_main():
    create_directory_structure("uc03featureselection")
    # Run the script using subprocess
    run_pytest()
    # run_data_processing()
    features = run_data_processing()
    run_train()
    run_evaluate()
    run_deploy()
    # run_inference()
    run_inference(features)
    run_monitor()


if __name__ == "__main__":
    run_main()
