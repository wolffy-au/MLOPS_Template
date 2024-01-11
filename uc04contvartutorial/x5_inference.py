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
    [model] = load_models(model_name, save_path="uc04contvartutorial/data/processed/")

    names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    data = [
        [
            8.3252,
            41.0,
            6.984126984126984,
            1.0238095238095237,
            322.0,
            2.5555555555555554,
            37.88,
            -122.23,
        ],  # 4.526
        [
            2.6042,
            46.0,
            4.489563567362429,
            1.0910815939278937,
            1647.0,
            3.1252371916508537,
            37.79,
            -122.22,
        ],  # 1.247
        [
            5.1498,
            35.0,
            7.256130790190736,
            1.0544959128065394,
            1086.0,
            2.9591280653950953,
            37.71,
            -122.09,
        ],  # 2.664
        [
            10.5941,
            18.0,
            7.537037037037037,
            0.9481481481481482,
            1580.0,
            2.925925925925926,
            35.34,
            -119.08,
        ],  # 2.458
        [
            8.5325,
            14.0,
            7.973584905660378,
            0.9660377358490566,
            842.0,
            3.177358490566038,
            35.35,
            -119.09,
        ],  # 2.241
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
