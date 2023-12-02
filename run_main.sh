#!/bin/sh

python src/1_data_processing.py
python src/2_train.py
python src/3_evaluate.py
python src/4_deploy.py
python src/5_inference.py
python src/6_monitor.py
