from uctemplate.mkdirs import create_directory_structure
from uctemplate.x1_data_processing import run_data_processing
from uctemplate.x2_train import run_train
from uctemplate.x3_evaluate import run_evaluate
from uctemplate.x4_deploy import run_deploy
from uctemplate.x5_inference import run_inference
from uctemplate.x6_monitor import run_monitor
from uctemplate.x7_test import run_pytest

def run_main():
    #create_directory_structure('.')
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
