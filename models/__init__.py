# Auto Generate init file

# models/ __init__.py

import os,joblib

def load_model(model_dir:str ,model_name:str):
    path = os.path.join(model_dir,model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model Not Found at {path}')
    else:
        return joblib.load(path)