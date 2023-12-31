# 5. Inference Script (inference.py):
# This script is used for making predictions or inferences using a deployed model.
# It may include code for handling input data, calling the model, and processing the output.
# Part of the deployment pipeline to ensure that the deployed model functions as expected.

from libmlops.models.model_loading import load_models
from libmlops.utils.features_evaluation import keep_features

def run_inference(features=[]):
    # print("Loading previous model")
    # model_name = "finalised_model"
    # [model] = load_models(model_name)

    # data = [
    #     [5.9, 4.1, 1.5, 0.3], # Iris-setosa
    #     [7.1, 3.3, 4.8, 1.5], # Iris-versicolor
    #     [7.1, 3.5, 6.0, 2.6], # Iris-virginica
    #     ]

    # data = [
    #     [6,148,72,35,0,33.6,0.627,50],
    #     [8,183,64,0,0,23.3,0.672,32],
    #     [0,137,40,35,168,43.1,2.288,33],
    #     [3,78,50,32,88,31.0,0.248,26],
    #     [2,197,70,45,543,30.5,0.158,53],
    #     [8,125,96,0,0,0.0,0.232,54],
    #     [10,168,74,0,0,38.0,0.537,34],
    #     [1,189,60,23,846,30.1,0.398,59],
    #     ]
        
    # if features != []:
    #     data = keep_features(data, features, keep_y=False)

    # for x in data:
    #     # test the model with 1 row
    #     print(model.predict([x]))
    pass

if __name__ == "__main__":
    run_inference()
