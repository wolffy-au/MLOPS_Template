import joblib as jl
import os

SAVE_MODEL_PATH = os.path.normpath("data/raw/")

def save_models(models, filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    for (model, filename) in zip(models, filenames):
        jl.dump(model, f"{save_path}{filename}.model")

def load_models(filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    models = []
    for filename in filenames:
        models.append(jl.load(f"{save_path}{filename}.model"))
    return models

def clear_models(filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    for filename in filenames:
        file_path = os.path.join(f"{save_path}{filename}.model")
        os.remove(file_path)
