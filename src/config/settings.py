import os
import numpy as np
import pandas as pd
import mlflow

from time import time
from typing import Any,Dict,Optional
from sklearn.model_selection import RandomizedSearchCV,train_test_split,PredefinedSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from .settings import get_settings
from .paths import get_path
from logger import logger


env_path = os.path.abspath(os.path.join(os.getcwd(),'..','..','.env'))
load_dotenv(dotenv_path=env_path)



class settings:
    def __init__(self) -> None:
        
        self.path = get_path()

        self.n_split = int(os.getenv('N_SPLIT',5))
        self.random_seed = int(os.getenv('RANDOM_SEED',42))
        self.early_stop = int(os.getenv('EARLY_STOPPING_ROUND','20'))


        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI','http://localhost:5000')
        if not self.mlflow_uri:
            raise ValueError('MLFLOW_TRACKING_URI is not set in environemnt or .env file')
        
        self.mlflow_experiment = os.getenv('MLFLOW_EXPERIMENT_NAME','Defaul_EXPERIMENT')


        self.xgb_params = {
            'n_estimators':    [100,300,500],
            'learning_rate':   [0.01,0.1,0.3],
            'max_depth':       [3,6,9],
            'reg_alpha':       [3,6,9],
            'reg_lambda':      [3,6,9],
            'subsample':       [0.3,0.6,0.8],
            'colsample_bytree':[0.3,0.6,0.8],
            'gamma':           [0,0.1,0.3],
            'scale_pos_weight':[1,50,99],
            'min_child_weight':[1,3,5]
        }

        self.lgb_params = {
            'n_estimators'    :[100,300,500],
            'learning_rate'   :[0.01,0.1,0.3],
            'max_depth'       :[3,6,9],
            'reg_alpha'       :[3,6,9],
            'reg_lambda'      :[3,6,9],
            'subsample'       :[0.01,0.1,0.3],
            'colsample_bytree':[0.01,0.1,0.3],
            'min_child_weight':[1,3,6],
            'num_leaves'      :[31,50,100]
        }

        self.cat_params = {
            'iterations'      :[100,300,500],
            'learning_rate'   :[0.01,0.1,0.3],
            'l2_leaf_reg'     :[1.0,5.0,10.0],
            'random_strength' :[1.0,5.0,10.0],
            'border_count'    :[32,64,128],
            'loss_function'   :['logloss'],
            'depth'           :[3,6,9]
        }

    def xgb_randomized_tunned(
                self,
                X_train:pd.DataFrame,
                y_train:pd.Series,
                scoring:Optional[str] = 'f1',
                eval_metric:Optional[str] = None
                ) -> Dict[str,Any]:
        
        X_tr,X_val,y_tr,y_val = train_test_split(
                X_train,y_train,
                random_state=self.random_seed,
                stratify=y_train,
                test_size=0.1
                )
        
        X_full = pd.concat((X_tr,X_val),axis=0)
        y_full = pd.concat((y_tr,y_val),axis=0)

        test_fold = [-1] * len(X_tr) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold)

        fit_params = {
            'early_stopping_rounds':self.early_stop,
            'eval_set':[(X_val,y_val)],
            'verbose': False
        }

        if eval_metric:
            fit_params['eval_metric'] = eval_metric


        model = XGBClassifier(
                eval_metric='logloss',
                random_state=self.random_seed
                )
        
        grid = RandomizedSearchCV(
                random_state=self.random_seed,
                param_distributions=self.xgb_params,
                estimator=model,
                scoring=scoring,
                n_jobs=-1,
                cv=ps
                )
        start = time()
        grid.fit(X_full,y_full,**fit_params)
        elapsed = time() - start

        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.set_tracking_uri(self.mlflow_uri)
        with mlflow.start_run('XGB_RandomSearch'):
            mlflow.log_param('XGB_Tunned_Params',grid.best_params_)
            mlflow.log_metric('Tunnin Time Sec:.',elapsed)
            mlflow.log_metric('Best Score',grid.best_score_)
            mlflow.set_tag('Model','XGBoost')
            
        return grid.best_params_
    
    def lgb_randomized_tune(
                self,
                X_train:pd.DataFrame,
                y_train:pd.Series,
                eval_metric:Optional[str] = None,
                scoring:Optional[str] = 'f1'
            ) -> Dict[str,Any]:
        
        X_tr,X_val,y_tr,y_val = train_test_split(
            X_train,y_train,
            test_size=0.1,
            random_state=self.random_seed,
            stratify=y_train
            )
        
        X_full = pd.concat((X_tr,X_val),axis=0)
        y_full = pd.concat((y_tr,y_val),axis=0)


        test_fold = [-1] * len(X_tr) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold)

        fit_params = {
            'early_stopping_rounds':self.early_stop,
            'eval_set':[(X_val,y_val)],
            'verbose':False
        }
        if eval_metric:
            fit_params['eval_metrix'] = eval_metric
        
        model:Any = LGBMClassifier(
            random_state=self.random_seed
            )
        
        grid = RandomizedSearchCV(
            param_distributions=self.lgb_params,
            estimator=model,
            scoring=scoring,
            n_jobs=-1,
            cv=ps,
            random_state=self.random_seed
            )
        start = time()
        grid.fit(X_full,y_full,**fit_params)
        end = time() - start

        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.set_tracking_uri(self.mlflow_uri)
        with mlflow.start_run(run_name='LGB_RandomSearch'):
            mlflow.log_metric('Tunning Time Sec:.',end)
            mlflow.log_metric('Best Score',grid.best_score_)
            mlflow.log_param('LGB_Params',grid.best_params_)
            mlflow.set_tag('Model','LGBoost')
        return grid.best_params_
    def cat_randomized_tune(
            self,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            scoring:Optional[str] = 'f1'
    ) -> dict:
        
        X_tr,X_val,y_tr,y_val = train_test_split(
            X_train,y_train,
            random_state=self.random_seed,
            stratify=y_train,
            test_size=0.1,
            )
        
        X_full = pd.concat((X_tr,X_val),axis=0)
        y_full = pd.concat((y_tr,y_val),axis=0)

        test_fold = [-1] * len(X_tr) + [0] * len(X_val)

        fit_params = {
            'early_stopping_rounds':20,
            'eval_set':[(X_val,y_val)],
            'verbose':False
        }

        ps = PredefinedSplit(test_fold)

        model:Any = CatBoostClassifier(
            random_state=self.random_seed
        )
        
        grid = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.cat_params,
            scoring=scoring,
            n_jobs=-1,
            cv=ps
        )
        
        start = time()
        grid.fit(X_full,y_full,**fit_params)
        end = time() - start

        for param_name,param_val in grid.best_params_.items():
            mlflow.log_param(param_name,param_val)

        mlflow.log_metric('Tunning Time',end)
        mlflow.log_metric('Best Score',grid.best_score_)
        mlflow.set_tag('Model','Cat')
        return grid.best_params_
    
def get_settings():
    return settings()