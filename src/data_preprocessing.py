import os
import pandas as pd, numpy as np
from typing import List,Tuple
from sklearn.model_selection import StratifiedKFold
from config import conf

def load_csv(path:str,cols:List[str]) -> Tuple[pd.DataFrame,pd.Series]:
    df = pd.read_csv(path)

    df['Hour'] = (df['Time'] // 3600 ) % 24
    df['Day'] = (df['Time'] // (3600 * 24))
    df['LogAmount'] = np.log1p(df['Amount'])

    y = df['Class']
    X = df.drop(columns=cols)
    return X,y


def download_data(drop_cols):
    file_path = conf.data_csv_path

    if not os.path.exists(file_path):
        print('Downloading Dataset using kaggle API ...')
        os.system(f'kaggle datasets download -d forsong/credit-card-fraud-detection -p {conf.data_path}')
        zip_path = os.path.join(conf.data_path,'credit-card-fraud-detection.zip')
        if os.path.exists(zip_path):
            os.system(f'unzip -o {zip_path} -d {conf.data_path}')
            os.remove(zip_path)
        print('Download Complete')
    return load_csv(conf.data_path,drop_cols)


def preprocess_kf(X:pd.DataFrame,y:pd.Series) -> Tuple[
    List[pd.DataFrame],List[pd.Series],List[pd.DataFrame],List[pd.Series]
    ]:
    for col in X.columns:

        X[col] = X[col].fillna(X[col].mean())
    y = y.fillna(y.mode()[0])
    
    X_tr,y_tr,X_te,y_te = [],[],[],[]
    
    kf = StratifiedKFold(n_splits=conf.n_split,random_state=conf.random_seed,shuffle=True)
    
    for train_idx,test_idx in kf.split(X,y):
        X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
        X_test,y_test = X.iloc[test_idx],y.iloc[test_idx]

        X_tr.append(X_train)
        X_te.append(X_test)
        y_tr.append(y_train)
        y_te.append(y_test)

    return X_tr,y_tr,X_te,y_te
