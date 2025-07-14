import os
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from typing import List,Tuple,cast
from imblearn.combine import SMOTEENN
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit
from config.paths import get_path
from config.settings import get_settings
conf = get_path()
setting = get_settings()

class preprocess:
    def __init__(self) -> None:
        pass

    def load_csv(self,path:str,cols:List[str]) -> Tuple[pd.DataFrame,pd.Series]:

        df = pd.read_csv(path)

        df['Hour'] = (df['Time'] // 3600 ) % 24
        df['Day'] = (df['Time'] // (3600 * 24))
        df['LogAmount'] = np.log1p(df['Amount'])

        y = df['Class']
        X = df.drop(columns=cols)
        return X,y


    def download_data(self,drop_cols):
        file_path = conf.DATA_CSV_PATH

        if not os.path.exists(file_path):
            print('Downloading Dataset using kaggle API ...')
            os.system(f'kaggle datasets download -d forsong/credit-card-fraud-detection -p {conf.DATA_PATH}')
            zip_path = os.path.join(conf.DATA_PATH,'credit-card-fraud-detection.zip')
            if os.path.exists(zip_path):
                os.system(f'unzip -o {zip_path} -d {conf.DATA_PATH}')
                os.remove(zip_path)
            print('Download Complete')
        X,y = self.load_csv(conf.DATA_PATH,drop_cols)
        return X,y

    def to_dense_array(self,X):
        if issparse(X):
            return X.toarray()
        elif isinstance(X,pd.DataFrame):
            return X.values
        elif isinstance(X,pd.Series):
            return X.to_numpy().reshape(-1,1)
        elif isinstance(X,np.ndarray):
            return X.reshape(-1,1) if X.ndim == 1 else X
        else:
            raise TypeError(f'Unsupported Type for conversion to dense array {type(X)}')

    def preprocess_kf(
            self,
                X:pd.DataFrame,
                y:pd.Series,
                to_dense:bool = False
        ) -> Tuple[
            List[pd.DataFrame],List[pd.Series],
            List[pd.DataFrame],List[pd.Series]
        ]:

        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
        y = y.fillna(y.mode()[0])

        X_tr,y_tr,X_te,y_te = [],[],[],[]

        kf = StratifiedKFold(setting.n_split,shuffle=True,random_state=setting.random_seed)

        for train_idx,test_idx in kf.split(X,y):
            X_train,y_train = X.iloc[train_idx],y.iloc[test_idx]
            X_test, y_test  = X.iloc[train_idx], y.iloc[test_idx]

            sm = SMOTEENN(random_state=setting.random_seed,n_jobs=-1)
            resampled = sm.fit_resample(X_train,y_train)
            
            if len(resampled) == 2:
                X_res,y_res = resampled
                        
            elif len(resampled) > 2:
                        
                X_res,y_res = resampled[0],resampled[1]
                extra_element = resampled[2:]
                print("Extra Element Got During SMOTEENN: ", extra_element)
                
            if to_dense:
                
                X_res = self.to_dense_array(X_res)
                X_test = self.to_dense_array(X_test)
                y_res = self.to_dense_array(y_res).ravel()
                y_test = self.to_dense_array(y_test).ravel()
                
                X_res = pd.DataFrame(X_res,columns=X_train.columns)
                X_test = pd.DataFrame(X_test,columns=X_train.columns)
                y_res = pd.Series(y_res)
                y_test = pd.Series(y_test)
            
            X_tr.append(X_res)
            y_tr.append(y_res)
            X_te.append(X_test)
            y_te.append(y_test)
        return X_tr,y_tr,X_te,y_te




    def preprocess_tss(
            self,
            X:pd.DataFrame,
            y:pd.Series,
            to_dense:bool = False
            ) ->Tuple[
        List[pd.DataFrame],List[pd.Series],
        List[pd.DataFrame],List[pd.Series]
        ]:
        
        
        for col in X.columns:
            X[col] = X[col].fillna(X[col].mean())
        y = y.fillna(y.mode()[0])

        X_tr,y_tr,X_te,y_te = [],[],[],[]
        
        tss = TimeSeriesSplit(setting.n_split)

        for train_idx,test_idx in tss.split(X,y):
            X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
            X_test,y_test =  X.iloc[test_idx],y.iloc[test_idx]

            sm = SMOTEENN(random_state=setting.random_seed,n_jobs=-1)
            resampled = sm.fit_resample(X_train,y_train)

            if len(resampled) == 2:
                X_res,y_res = resampled
            elif len(resampled) > 2:
                X_res,y_res = resampled[0],resampled[1]
                print(f'Extra Elements Got During SMOTEENN: {resampled[:2]}')
            
            if to_dense:
                y_res = self.to_dense_array(y_res)
                X_res = self.to_dense_array(X_res)
                X_test = self.to_dense_array(X_test)
                y_test = self.to_dense_array(y_test)

                X_res = pd.DataFrame(X_res)
                y_res = pd.Series(y_res)
                X_test = pd.DataFrame(X_test)
                y_test = pd.Series(y_test)
                
            X_tr.append(X_res)
            y_tr.append(y_res)
            X_te.append(X_test)
            y_te.append(y_test)

        return X_tr,y_tr,X_te,y_te

def get_preprocess():
    return preprocess()