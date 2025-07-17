# Standard
import os
from time import time
# Third-party

import joblib
import mlflow
import optuna
import traceback
import numpy as np
import pandas as pd
from typing import Any,Optional,Literal,cast,Tuple
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# Internal modules
from src.logger import logger
from src.config.paths import get_path
from src.config.settings import get_settings
from src.data_preprocessing import get_preprocess


preprocess = get_preprocess()
paths = get_path()
settings = get_settings()

all_f1,all_roc,all_results = [],[],[]
all_clf,all_cm = [],[]

tuner_typer = Optional[Literal['optuna','randomized']]
class train:
    def __init__(self) -> None:
        pass

    def train_xgb(self,tunner:tuner_typer = 'randomized'):
        try:
            cols = ['Time','Amount','Class']
            X,y = preprocess.load_csv(paths.DATA_CSV_PATH,cols)
        except:
            cols = ['Time','Amount','Class']
            X,y = preprocess.download_data(cols)

        try:
            X_tr,y_tr,X_te,y_te = preprocess.preprocess_kf(X,y)

        except ValueError as val_err:

            logger.error(f'Value error accured during preprocessing of data: {val_err}')

        mlflow.set_experiment(settings.mlflow_experiment_name)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        logger.info(f'Starting The Training Of XGB model ...')
        try:

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te),start=1):
                
                try:
                    mlflow.start_run(run_name=f'Train XGB model with Fold_{fold}',nested=True)

                    logger.info(f'Training model / Fold: {fold} using Randomizedcv for tuning')

                    if tunner == 'randomized':
                        params = settings.xgb_randomized_search(X_train,y_train)
                    
                    elif tunner == 'optuna':
                        def wraped(trial):
                            return settings.xgb_optuna_tune(trial,X_train,y_train)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(wraped,n_jobs=-1,n_trials=30)
                        params = study.best_params
                    

                    model = XGBClassifier(
                        **params,
                        eval_metric='logloss',
                        random_state=settings.random_seed_value
                        )
                    model.fit(X_train,y_train)
                    
                    preds = model.predict(X_test)
                    proba = model.predict_proba(X_test)[:,1]

                    f1 = f1_score(y_test,preds)
                    roc = roc_auc_score(y_test,preds)
                    cm = confusion_matrix(y_test,preds)
                    clf = classification_report(y_test,preds)

                    all_f1.append(f1)
                    all_roc.append(roc)
                    all_cm.append(cm)
                    all_clf.append(clf)
                    all_results.append({f'Fold':fold,'y_test':y_test,'proba':proba})
                    

                    mlflow.log_metric('F1_Score',float(f1))
                    mlflow.log_metric('Roc_Auc_Score',float(roc))
                    mlflow.log_dict({'XGB_CM':cm.tolist()},'cm.json')

                    fold_report_path = os.path.join(paths.REPORT_PATH,f'CLF_Fold_{fold}.txt')
                    with open(fold_report_path,'w') as f:
                        f.write(str(clf))

                    fold_model_path = os.path.join(paths.XGB_PATH,f'XGB_Fold_{fold}.joblib')
                    joblib.dump(model,fold_model_path)
                    mlflow.log_artifact(fold_model_path)

                    logger.info(f'Fold_{fold} XGB_model Saved to {fold_model_path}')
                except Exception as e:
                    logger.error(f' XGB Training Failed at Fold {fold} due to {e}')
        except FileNotFoundError as file_err:
            logger.critical(f'Data File Not Found: {file_err}')
        except ValueError as val_err:
            logger.error(f'Value error During Training/Preprocessing: {val_err}')
        except ConnectionError as conn_err:
            logger.error(f'Failed To Connect to MLFlow Server: {conn_err}')
        except Exception as e:
            logger.error(f'An unexpected error accured during model Training: {e}')
        return all_f1,all_roc,all_cm,all_clf,all_results
    
    def train_lgb(self,tunner:tuner_typer = 'randomized'):
        try:
            
            cols = ['Class','Amount','Time']
            X,y = preprocess.load_csv(paths.DATA_CSV_PATH,cols=cols)

        except:
            
            cols = ['Class','Amount','Time']
            X,y = preprocess.download_data(drop_cols=cols)

        try:
            
            X_tr,y_tr,X_te,y_te = preprocess.preprocess_kf(X,y,to_dense=True)

            
        except SyntaxError as snx_err:
            logger.info(f'error accured during preprocessing of data: {snx_err}')

        try:
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te),start=1):
                try:
                    mlflow.start_run(run_name=f'Training LGB model with Fold_{fold}',nested=True)

                    if tunner == 'randomized':

                        params = settings.lgb_randomized_search(X_train,y_train)
                    
                    elif tunner == 'optuna':
                        def wraped(trial):
                            return settings.lgb_optuna_tune(trial,X_train,y_train)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(wraped,n_jobs=-1,n_trials=30)
                        params = study.best_params
                    

                    model:Any = LGBMClassifier(**params)
                    model.fit(X_train,y_train)

                    preds = model.predict(X_test)
                    preds = cast(np.ndarray,preds)
                    proba = model.predict_proba(X_test)[:,1] 

                    f1 = f1_score(y_test,preds) 
                    roc = roc_auc_score(y_test,preds) 
                    cm = confusion_matrix(y_test,preds) 
                    clf = classification_report(y_test,preds) 

                    all_f1.append(f1)
                    all_roc.append(roc)
                    all_cm.append(cm)
                    all_clf.append(clf)
                    all_results.append({'Fold':fold,'y_test':y_test,'proba':proba})

                    
                    fold_report_path = os.path.join(paths.REPORT_PATH,f'CM_Fold_{fold}.txt')
                    with open(fold_report_path,'w') as f:
                        f.write(str(clf))

                    fold_model_path = os.path.join(paths.LGB_PATH,f'LGB_Fold_{fold}.joblib')
                    joblib.dump(model,fold_model_path)

                    
                    mlflow.log_metric('F1_Score',float(f1))
                    mlflow.log_metric('Roc_Auc',float(roc))

                    mlflow.log_dict({f'CM_{fold}':cm.tolist()},'cm.json')
                    mlflow.log_artifact(fold_report_path)
                    mlflow.log_artifact(fold_model_path)
                    

                except Exception as e:
                    logger.error(f'error accured on Fold_{fold} as {e}')
        except ValueError as val_err:
            logger.error(f'Model Training/Prediction Failed as {val_err}')
        except FileNotFoundError as file_err:
            logger.error(f'Data File Not Found: {file_err}')
        except ConnectionError as conn_err:
            logger.error(f'Can not Conentect to MLFlow Server: {conn_err}')
        return all_f1,all_roc,all_cm,all_clf,all_results
    
    def train_cat(self,tunner:tuner_typer = 'randomized'):
        try:
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            try:
                
                cols = ['Class','Amount','Time']
                X,y = preprocess.load_csv(paths.DATA_CSV_PATH,cols=cols)

            except:
                
                cols = ['Class','Amount','Time']
                X,y = preprocess.download_data(drop_cols=cols)
            try:
                X_tr,y_tr,X_te,y_te = preprocess.preprocess_kf(X,y)

            except Exception as e:
                logger.error(f'Unexpected error Occured During Preprocessing of Data: {e}')
            
            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te),start=1):
                try:
                    with mlflow.start_run(run_name=f'Training CatBooster Fold_{fold}',nested=True):

                        if tunner == 'randomized':
                            params = settings.catboost_randomized_search(X_train,y_train)


                        elif tunner == 'optuna':
                            def wraped(trial):
                                return settings.cat_optuna_tune(trial,X_train,y_train)
                            
                            study = optuna.create_study(direction='maximize')
                            study.optimize(wraped,n_jobs=-1,n_trials=30)
                            params = study.best_params
                    

                        model = CatBoostClassifier(**params)
                        model.fit(X_train,y_train)

                        preds = model.predict(X_test)
                        proba = model.predict_proba(X_test)[:,1]

                        f1 = f1_score(y_test,preds)
                        roc = roc_auc_score(y_test,preds)
                        cm = confusion_matrix(y_test,preds)
                        clf = classification_report(y_test,preds)

                        all_f1.append(f1)
                        all_roc.append(roc)
                        all_cm.append(cm)
                        all_clf.append(clf)
                        all_results.append({'Fold':fold,'y_test':y_test,'proba':proba})


                        fold_report_path = os.path.join(paths.REPORT_PATH,f'Cat_Report_Fold_{fold}.txt')
                        with open(fold_report_path,'w') as f:
                            f.write(str(clf))
                        
                        fold_model_path = os.path.join(paths.CAT_PATH,f'Cat_Model_Fold_{fold}.joblib')
                        joblib.dump(model,fold_model_path)


                        mlflow.log_metric('F1_Score',float(f1))
                        mlflow.log_metric('Roc_Auc_Score',float(roc))


                        mlflow.log_dict({'ConFusion_Matrix':cm},'cm.json')
                        mlflow.log_artifact(fold_report_path)
                        mlflow.log_artifact(fold_model_path)
                        

                except Exception as e:
                    logger.error(f'UnExpected error on Training Cat_Model Fold_{fold}')
                    traceback.print_exc()
        except FileNotFoundError as file_err:
            logger.error(f'File Not Found For Training: {file_err}')
        except ValueError as val_err:
            logger.error(f'Model Training/Preprocessing Error: {val_err}')
        except ConnectionError as conn_err:
            logger.error(f'Connection Timeout While Connecting to mlflow Server: {conn_err}')

        return all_f1,all_roc,all_cm,all_clf,all_results
    
    def train_voting(self,tunner:tuner_typer = 'randomized'):
        try:
            cols = ['Time','Class','Amount']
            X,y = preprocess.load_csv(paths.DATA_CSV_PATH,cols)
            
            logger.info(f'Successfully Loaded CSV from: {paths.DATA_CSV_PATH}')
        except:
            cols = ['Time','Class','Amount']
            X,y = preprocess.download_data(cols)

            logger.warning(f'Failed To Load CSV from: {paths.DATA_CSV_PATH}')
            logger.info(f'Downloading CSV from Kaggle Api')

        try:
            X_tr,y_tr,X_te,y_te = preprocess.preprocess_kf(X,y)
        except Exception as e:
            logger.error(f'Unexpected Error occured durin Preprocesing Data: {e}')
        
        try:
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te)):

                with mlflow.start_run(run_name=f'Starting Training by Fold_{fold} of Voting Classifier',nested=True):

                    if tunner  == 'randomized':
                        votting = settings.voting_ranomized_search(X_train,y_train)
                    
                    elif tunner == 'optuna':
                        votting = settings.votting_optuna_tune(X_train,y_train)

                    voting_start = time()
                    

                    votting.fit(X_train,y_train)

                    voting_time = time() - voting_start

                    logger.info(f'Voting model Fitting Time: {voting_time}')


                    preds = votting.predict(X_test)
                    proba = votting.predict_proba(X_test)[:,1]

                    f1 = f1_score(y_test,preds)
                    roc = roc_auc_score(y_test,preds)
                    cm = confusion_matrix(y_test,preds)
                    clf = classification_report(y_test,preds)

                    all_f1.append(f1)
                    all_roc.append(roc)
                    all_cm.append(cm)
                    all_clf.append(clf)
                    all_results.append({'Fold':fold,'y_test':y_test,'proba':proba})

                    fold_report_path = os.path.join(paths.REPORT_PATH,f'Fold_Report_{fold}.txt')
                    with open(fold_report_path,'w') as f:
                        f.write(str(clf))
                    
                    fold_model_path = os.path.join(paths.VOTNG_PATH,f'Fold_Voting_{fold}.joblib')
                    joblib.dump(votting,fold_model_path)

                    mlflow.log_metric('F1_Score',float(f1))
                    mlflow.log_metric('Roc_Auc',float(roc))

                    mlflow.log_artifact(fold_model_path)
                    mlflow.log_artifact(fold_model_path)
                    mlflow.log_dict({'Confusion_Matrix':cm},'cm.json')
                    mlflow.set_tag('Model','Voting')
            logger.info(f'Traning For VotingVlasssifier Completed')
        except Exception as e:
            logger.error(f'An unexpected error occured during train of voting model: {e}')
        return all_f1,all_roc,all_cm,all_clf,all_results
    

def get_model_training():
    return train()