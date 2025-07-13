import os,sys
import pandas as pd, numpy as np
from typing import List,Tuple
from sklearn.model_selection import StratifiedKFold
from config import conf

def load_csv(path:str,cols:List[str]) -> Tuple[pd.DataFrame,pd.Series]:
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    df['Hour'] = (df['Time'] // 3600 ) % 24
    df['Day'] = (df['Time'] // (3600 * 24))
    df['LogAmount'] = np.log1p(df['Amount'])

    if not cols:

        return df,pd.Series(dtype='float64')
    else:

        X = df.drop(columns=cols)
        y = df['Class']
        return X,y

def preprocess_kf(X:pd.DataFrame,y:pd.Series) -> Tuple[
    List[pd.DataFrame],List[pd.Series],List[pd.DataFrame],List[pd.Series]
    ]:
    for col in X.columns:

        X[col] = X[col].fillna(X[col].mean())
    y = y.fillna(y.mean())
    
    X_tr,y_tr,X_te,y_te = [],[],[],[]
    
    kf = StratifiedKFold(n_splits=conf.n_split,random_state=conf.random_seed)
    
    for train_idx,test_idx in kf.split(X,y):
        X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
        X_test,y_test = X.iloc[test_idx],y.iloc[test_idx]

        X_tr.append(X_train)
        X_te.append(X_test)
        y_tr.append(y_train)
        y_te.append(y_test)

    return X_tr,y_tr,X_te,y_te
