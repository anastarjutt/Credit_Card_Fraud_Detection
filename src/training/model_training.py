# Standard
import os

# Third-party
import joblib
import mlflow
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
        except FileNotFoundError as file_err:
            logger.error(f'File Not found during loading of data: {file_err}')
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

                    mlflow.log_metric('F1_Score',float(f1))
                    mlflow.log_metric('Roc_Auc',float(roc))

                    mlflow.log_dict({f'CM_{fold}':cm.tolist()},'cm.json')
                    
                    fold_report_path = os.path.join(paths.REPORT_PATH,f'CM_Fold_{fold}.txt')
                    with open(fold_report_path,'w') as f:
                        f.write(str(clf))

                    fold_model_path = os.path.join(paths.LGB_PATH,f'LGB_Fold_{fold}.joblib')
                    joblib.dump(model,fold_model_path)

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
                    with mlflow.start_run(run_)

def get_model_training():
    return train()