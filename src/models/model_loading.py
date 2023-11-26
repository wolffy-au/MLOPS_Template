import joblib as jl
import os

SAVE_MODEL_PATH = os.path.normpath("data/raw/")

def save_models(models, filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    if not isinstance(models, list):
        models = [models]
    if isinstance(filenames, str):
        filenames = [filenames]
    for (model, filename) in zip(models, filenames):
        model_path = os.path.join(save_path, f"{filename}.model")
        jl.dump(model, model_path)

def load_models(filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    if isinstance(filenames, str):
        filenames = [filenames]
    models = []
    for filename in filenames:
        model_path = os.path.join(save_path, f"{filename}.model")
        if os.path.exists(model_path):
            models.append(jl.load(model_path))
            
    return models

def clear_models(filenames, save_path=SAVE_MODEL_PATH):
    save_path = os.path.normpath(save_path)
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        model_path = os.path.join(save_path, f"{filename}.model")
        if os.path.exists(model_path):
            os.remove(model_path)
