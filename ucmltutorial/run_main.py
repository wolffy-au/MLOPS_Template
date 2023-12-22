from ucmltutorial.mkdirs import create_directory_structure
from ucmltutorial.x1_data_processing import run_data_processing
from ucmltutorial.x2_train import run_train
from ucmltutorial.x3_evaluate import run_evaluate
from ucmltutorial.x4_deploy import run_deploy
from ucmltutorial.x5_inference import run_inference
from ucmltutorial.x6_monitor import run_monitor
from ucmltutorial.x7_test import run_pytest

def run_main():
    create_directory_structure('ucmltutorial')
    # Run the script using subprocess
    # run_pytest()
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
