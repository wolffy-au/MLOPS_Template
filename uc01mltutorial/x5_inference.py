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
    [model] = load_models(model_name, "uc01mltutorial/data/processed/")

    names = [
        "sepal-length",
        "sepal-width",
        "petal-length",
        "petal-width",
    ]
    data = [
        [5.9, 4.1, 1.5, 0.3],  # Iris-setosa
        [7.1, 3.3, 4.8, 1.5],  # Iris-versicolor
        [7.1, 3.5, 6.0, 2.6],  # Iris-virginica
    ]
    df = pd.DataFrame(data, columns=names)

    if features != []:
        df = keep_features(df, features)
        print(features, df.columns)

    for index, row in df.iterrows():
        # test the model with 1 row
        print(model.predict(pd.DataFrame([row.values], columns=row.index)))
    pass


if __name__ == "__main__":
    run_inference()
