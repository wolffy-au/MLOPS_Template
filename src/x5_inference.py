# 5. Inference Script (inference.py):
# This script is used for making predictions or inferences using a deployed model.
# It may include code for handling input data, calling the model, and processing the output.
# Part of the deployment pipeline to ensure that the deployed model functions as expected.

from models.model_loading import load_models

def run_inference():
    # print("Loading previous model")
    # model_name = "finalised_model"
    # [model] = load_models(model_name)

    # data = [
    #     [7.9, 4.4, 6.9, 2.5],
    #     [7.3, 3.5, 2.5, 0.2],
    #     [7.3, 3.5, 2.5, 0.3],
    #     ]

    # for x in data:
    #     # test the model with 1 row
    #     print(model.predict([x]))
    pass

if __name__ == "__main__":
    run_inference()
