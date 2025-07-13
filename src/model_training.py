import os,joblib,traceback
import numpy as np
import pandas as pd
import mlflow
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,classification_report
from src.logger import logger
from src.config import conf
from src.data_preprocessing import load_csv,preprocess_kf

def train_XGB():
    all_xgb_f1,all_xgb_roc,all_xgb_cm = [],[],[]
    all_xgb_clf,all_xgb_results = [],[]

    try:
        cols = ['Time','Amount','Class']
        X,y = load_csv(conf.data_path,cols)

        X_tr,X_te,y_tr,y_te = preprocess_kf(X,y)
        
        for fold,(X_train,X_test,y_train,y_test) in enumerate(zip(X_tr,X_te,y_tr,y_te)):
            logger.info(f'Training XGB Model with {fold+1} on using StratifiedKFold Spliting')

            mlflow.set_tracking_uri(str(conf.mlflow_uri))
            mlflow.set_experiment(experiment_name='ML_Training')
            with mlflow.start_run(run_name=f'Fold_{fold+1}_XGB'):
                try:
                    random_params = conf.randomized_tune(conf.params,X_train,y_train)

                    xg = XGBClassifier(**random_params,
                                        eval_metric='logloss',
                                        use_label_encoder=False,
                                        random_state=conf.random_seed)
                    
                    xg_preds = xg.predict(X_test)
                    xg_proba = xg.predict_proba(X_test)[:,1]

                    f1 = f1_score(y_test,xg_preds)
                    roc = roc_auc_score(y_test,xg_preds)
                    cm = confusion_matrix(y_test,xg_preds)
                    clf = classification_report(y_test,xg_preds)

                    all_xgb_roc.append(roc)
                    all_xgb_clf.append(clf)
                    all_xgb_f1.append(f1)
                    all_xgb_cm.append(cm)
                    all_xgb_results.append({'Fold':fold+1,'y_test':y_test,'proba':xg_proba})
                    

                    report_path = os.path.join(conf.reports_path,f'classification_report_fold_{fold+1}.txt')
                    with open(report_path,'w') as f:
                        f.write(str(clf))
                    mlflow.log_metric('F1-Score',float(f1))
                    mlflow.log_metric('ROC',float(roc))
                    mlflow.log_dict({'Confusion_Matrix':cm.tolist()},'cm.json')

                    fold_model_path = os.path.join(conf.xgb_path,f'xgb_fold_{fold+1}.joblib')
                    joblib.dump(xg,fold_model_path)
                    mlflow.log_artifact(fold_model_path)
                    logger.info(f'Fold_{fold+1} xgb model saved on path {fold_model_path}')
                except Exception as fold_err:
                    logger.error(f'[Fold_{fold+1}] Training Failed: {fold_err}')
                    logger.debug(traceback.format_exc())
                    raise
                
    except FileNotFoundError as fnf_err:
        logger.critical(f'Data File Not Found: {fnf_err}')
        raise
    except ValueError as val_err:
        logger.error(f'Value Error During Preprocessing or Training: {val_err}')
        raise
    except ConnectionError as conn_err:
        logger.error(f'Failed To Connect To MLFlow Server')
        raise
    except Exception as e:
        logger.error(f'An unExpected Error occured during model Training: {e}')
    
