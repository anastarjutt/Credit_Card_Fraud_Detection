# Standard
import os
from time import time
# Third-party
import joblib
import mlflow
from typing import Any
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# Internal modules
from logger import logger
from src.config.paths import get_path
from src.config.settings import get_settings
from data_preprocessing import get_preprocess


preprocess = get_preprocess()
paths = get_path()
settings = get_settings()

all_f1,all_roc,all_results = [],[],[]
all_clf,all_cm = [],[]

class train:
    def __init__(self) -> None:
        pass

    def train_xgb(self):
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

        mlflow.set_experiment(settings.mlflow_experiment)
        mlflow.set_tracking_uri(settings.mlflow_uri)
        logger.info(f'Starting The Training Of XGB model ...')
        try:

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te),start=1):
                
                try:
                    mlflow.start_run(run_name=f'Train XGB model with Fold_{fold}')

                    logger.info(f'Training model / Fold: {fold} using Randomizedcv for tuning')

                    params = settings.xgb_randomized_tunned(X_train,y_train)

                    model = XGBClassifier(
                        **params,
                        eval_metric='logloss',
                        use_label_encoder=False,
                        random_state=settings.random_seed
                        )
                    
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
    
    def train_lgb(self):
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
            mlflow.set_experiment(settings.mlflow_experiment)
            mlflow.set_tracking_uri(settings.mlflow_uri)

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te),start=1):
                try:
                    mlflow.start_run(run_name=f'Training LGB model with Fold_{fold}')


                    best_params = settings.lgb_randomized_tune(X_train,y_train,eval_metric='logloss')
                    model = LGBMClassifier(**best_params)

                    preds = model.predict(X_test)
                    proba = model.predict_proba(X_test)[:,1] # type: ignore

                    f1 = f1_score(y_test,preds) # type: ignore
                    roc = roc_auc_score(y_test,preds) # type: ignore
                    cm = confusion_matrix(y_test,preds) # type: ignore
                    clf = classification_report(y_test,preds) # type: ignore

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
    
    def train_cat(self):
        try:
            mlflow.set_experiment(settings.mlflow_experiment)
            mlflow.set_tracking_uri(settings.mlflow_uri)
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
                    with mlflow.start_run(run_name=f'Training CatBooster Fold_{fold}'):


                        params = settings.cat_randomized_tune(X_train,y_train)
                        model = CatBoostClassifier(**params)

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
        except FileNotFoundError as file_err:
            logger.error(f'File Not Found For Training: {file_err}')
        except ValueError as val_err:
            logger.error(f'Model Training/Preprocessing Error: {val_err}')
        except ConnectionError as conn_err:
            logger.error(f'Connection Timeout While Connecting to mlflow Server: {conn_err}')

        return all_f1,all_roc,all_cm,all_clf,all_results
    
    def train_voting(self):
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
            mlflow.set_experiment(settings.mlflow_experiment)
            mlflow.set_tracking_uri(settings.mlflow_uri)

            for fold,(X_train,y_train,X_test,y_test) in enumerate(zip(X_tr,y_tr,X_te,y_te)):

                with mlflow.start_run(run_name=f'Starting Training by Fold_{fold} of Voting Classifier'):

                    xgb_start = time()
                    xgb_params = settings.xgb_randomized_tunned(X_train,y_train)
                    xg = XGBClassifier(
                        **xgb_params,
                        random_state=settings.random_seed
                        )
                    xgb_time = time() - xgb_start
                    
                    logger.info(f'Voting (XGB) model Tuning Time: {xgb_time}')

                    lgb_start = time()
                    lgb_params = settings.lgb_randomized_tune(X_train,y_train)
                    lg:Any = LGBMClassifier(
                        **lgb_params,
                        random_state=settings.random_seed
                        )
                    lgb_time = time() - lgb_start 

                    logger.info(f'Voting (LGB) model Tuning Time: {lgb_time}')


                    cat_start = time()
                    cat_params = settings.cat_randomized_tune(X_train,y_train)
                    cat:Any = CatBoostClassifier(
                        **cat_params,
                        random_state=settings.random_seed
                    )
                    cat_time = time() - cat_start

                    logger.info(f'Voting (Cat) model Tuning Time: {cat_time}')


                    voting_start = time()
                    votting = VotingClassifier(
                        estimators=[
                            ('XGB',xg),
                            ('LGB',lg),
                            ('CAT',cat)
                        ],
                        voting='soft',
                        n_jobs=-1,
                    )

                    votting.fit(X_train,y_train)

                    voting_time = time() - voting_start

                    logger.info(f'Voting model Fitting Time: {voting_time}')


                    preds = votting.predict(X_test)
                    proba = votting.predict_proba(X_test)[:,0]

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