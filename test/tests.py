import os
import joblib
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score

from src.config import conf
from src.data_preprocessing import preprocess_kf,load_csv
from src.model_training import train_XGB


@pytest.fixture(scope='module')
def dataset() -> Tuple[pd.DataFrame,pd.Series]:
    cols = ['Time','Class','Amount']
    X,y = load_csv(conf.data_path,cols=cols)
    assert not X.empty, 'DataFrame Loading Failed'
    assert not y.empty, 'Target Loading Failed'
    return X,y

def test_preprocess_kf(dataset):
    X,y = dataset
    X_train,X_test,y_train,y_test = preprocess_kf(X,y)
    
    assert len(X_train) == len(X_test), 'Length Of DataFrame Changed During Preprocessing'
    assert all(len(tr) > 0 for tr in X_train), 'X_train is Empty due to KF Preprocessing'
    assert all(len(te) > 0 for te in X_test), 'X_test is Empty Due to KF Preprocessing'

def test_xgb_training(dataset,tmp_path):
    X,y = dataset
    xg = XGBClassifier(eval_metric='logloss',use_label_encoder=False,random_state=conf.random_seed)
    xg.fit(X.head(100),y.head(100))

    preds = xg.predict(X.head(10))
    f1 = f1_score(y.head(10),preds)

    assert len(preds) == 10
    assert isinstance(xg,BaseEstimator)
    assert set(preds).issubset({0,1})
    assert 0 >= f1 <= 1 ,'Invalid F1-Score'

    joblib.dump(xg,tmp_path)
    loaded = joblib.load(tmp_path)

    preds_loaded = loaded.predict(X.head(10))
    f1_loaded = f1_score(y.head(10),preds_loaded)

    assert len(preds) == len(preds_loaded),'loaded model prediction output changed'
    assert f1 ==  f1_loaded ,'Performance of Model decreased due to saving/loading by joblib'
    assert isinstance(loaded,XGBClassifier),'Loaded model nature changed'