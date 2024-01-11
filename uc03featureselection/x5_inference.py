# 5. Inference Script (inference.py):
# This script is used for making predictions or inferences using a deployed model.
# It may include code for handling input data, calling the model, and processing the output.
# Part of the deployment pipeline to ensure that the deployed model functions as expected.

import pandas as pd
from libmlops.models.model_loading import load_models
from libmlops.utils.features_evaluation import keep_features


def run_inference(features=[]):
    print("Loading previous model")
    model_name = "finalised_model"
    [model] = load_models(
        model_name,
        "uc03featureselection/data/processed",
    )

    names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"]
    data = [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],
        [8, 183, 64, 0, 0, 23.3, 0.672, 32],
        [0, 137, 40, 35, 168, 43.1, 2.288, 33],
        [3, 78, 50, 32, 88, 31.0, 0.248, 26],
        [2, 197, 70, 45, 543, 30.5, 0.158, 53],
        [8, 125, 96, 0, 0, 0.0, 0.232, 54],
        [10, 168, 74, 0, 0, 38.0, 0.537, 34],
        [1, 189, 60, 23, 846, 30.1, 0.398, 59],
    ]

    df = pd.DataFrame(data, columns=names)
    print(df.head())

    if features != []:
        df = keep_features(df, features)
        print(features, df.columns)
        print(df.head())

    for index, row in df.iterrows():
        # test the model with 1 row
        print(index, row.values)
        print(model.predict(pd.DataFrame([row.values], columns=row.index)))

    pass


if __name__ == "__main__":
    run_inference()
