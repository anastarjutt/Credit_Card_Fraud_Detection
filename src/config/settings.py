import os
import optuna
import numpy as np
import pandas as pd
import mlflow

from time import time
from typing import Any, Dict, Optional, Literal,cast,Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import VotingClassifier

import xgboost.callback as EarlyStopping
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from dotenv import load_dotenv
from src.config.paths import get_path
from src.logger import logger


env_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)


class Settings:
    def __init__(self) -> None:
        self.base_path = get_path()

        # General settings
        self.num_splits = int(os.getenv('N_SPLIT', 5))
        self.random_seed_value = int(os.getenv('RANDOM_SEED', 42))
        self.early_stopping_rounds = int(os.getenv('EARLY_STOPPING_ROUND', '20'))
        self.early_stopping = EarlyStopping.EarlyStopping(rounds=self.early_stopping_rounds, save_best=True) 
        self.tuning_method: Optional[Literal['optuna', 'randomized']] = 'randomized'

        # MLFlow settings
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        if not self.mlflow_tracking_uri:
            raise ValueError('MLFLOW_TRACKING_URI is not set in environment or .env file')
        
        self.mlflow_experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Default_EXPERIMENT')

        # XGBoost Hyperparameters
        self.xgb_hyperparameters = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9],
            'reg_alpha': [0,1,3],
            'reg_lambda': [0,1,3,],
            'subsample': [0.6, 0.8,1.0],
            'colsample_bytree': [0.6, 0.8,1.0],
            'gamma': [0, 0.1, 0.3],
            'scale_pos_weight': [1, 50, 99],
            'min_child_weight': [1, 3, 5]
        }

        # LightGBM Hyperparameters
        self.lgb_hyperparameters = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 6, 9],
            'reg_alpha': [0,1,3],
            'reg_lambda': [0,1,3,],
            'subsample': [0.6,0.8,1.0],
            'colsample_bytree': [0.6,0.8,1.0],
            'min_child_weight': [1, 3, 6],
            'num_leaves': [31, 50, 100]
        }

        # CatBoost Hyperparameters
        self.catboost_hyperparameters = {
            'iterations': [100, 300, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'l2_leaf_reg': [1.0, 5.0, 10.0],
            'random_strength': [1.0, 5.0, 10.0],
            'border_count': [32, 64, 128],
            'loss_function': ['Logloss'],
            'depth': [3, 6, 9]
        }

    def xgb_randomized_search(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            scoring_method: Optional[str] = 'f1',
            evaluation_metric: Optional[str] = None
    ) -> Dict[str, Any]:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            random_state=self.random_seed_value,
            stratify=y_train,
            test_size=0.1
        )

        full_X_data = pd.concat((X_train_split, X_val_split), axis=0)
        full_y_data = pd.concat((y_train_split, y_val_split), axis=0)

        predefined_split = [-1] * len(X_train_split) + [0] * len(X_val_split)
        predefined_split_object = PredefinedSplit(predefined_split)

        xgb_model = XGBClassifier(
            eval_metric='logloss',
            random_state=self.random_seed_value
        )

        random_search = RandomizedSearchCV(
            random_state=self.random_seed_value,
            param_distributions=self.xgb_hyperparameters,
            error_score='raise',
            estimator=xgb_model,
            scoring=scoring_method,
            n_jobs=-1,
            cv=predefined_split_object
        )

        start_time = time()
        random_search.fit(full_X_data, full_y_data)
        elapsed_time = time() - start_time
        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name='XGB_RandomSearch', nested=True):
            mlflow.log_metric('Tuning_Time_Seconds', elapsed_time)
            mlflow.log_param("dataset_used", "data/creditcard.csv")
            mlflow.log_metric('Best Score', random_search.best_score_)
            mlflow.set_tag('Model', 'XGBoost')

        return random_search.best_params_

    def lgb_randomized_search(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            evaluation_metric: Optional[str] = None,
            scoring_method: Optional[str] = 'f1'
    ) -> Dict[str, Any]:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.1,
            random_state=self.random_seed_value,
            stratify=y_train
        )

        full_X_data = pd.concat((X_train_split, X_val_split), axis=0)
        full_y_data = pd.concat((y_train_split, y_val_split), axis=0)

        predefined_split = [-1] * len(X_train_split) + [0] * len(X_val_split)
        predefined_split_object = PredefinedSplit(predefined_split)

        
        lgb_model: Any = LGBMClassifier(
            random_state=self.random_seed_value
        )

        random_search = RandomizedSearchCV(
            param_distributions=self.lgb_hyperparameters,
            estimator=lgb_model,
            error_score='raise',
            scoring=scoring_method,
            n_jobs=-1,
            cv=predefined_split_object,
            random_state=self.random_seed_value
        )

        start_time = time()
        random_search.fit(full_X_data, full_y_data)
        elapsed_time = time() - start_time

        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_name='LGB_RandomSearch', nested=True):
            mlflow.log_metric('Tuning_Time_Seconds', elapsed_time)
            mlflow.log_param("dataset_used", "data/creditcard.csv")
            mlflow.log_metric('Best_Score', random_search.best_score_)
            # Log each best param individually for better MLflow compatibility
            for param_name, param_value in random_search.best_params_.items():
                mlflow.log_param(f"LGB_{param_name}", param_value)
            mlflow.set_tag('Model', 'LightGBM')

        return random_search.best_params_

    def catboost_randomized_search(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            scoring_method: Optional[str] = 'f1'
    ) -> dict:
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            random_state=self.random_seed_value,
            stratify=y_train,
            test_size=0.1
        )

        full_X_data = pd.concat((X_train_split, X_val_split), axis=0)
        full_y_data = pd.concat((y_train_split, y_val_split), axis=0)

        predefined_split = [-1] * len(X_train_split) + [0] * len(X_val_split)

        predefined_split_object = PredefinedSplit(predefined_split)

        catboost_model:Any = CatBoostClassifier(
            random_state=self.random_seed_value
        )

        random_search = RandomizedSearchCV(
            estimator=catboost_model,
            param_distributions=self.catboost_hyperparameters,
            scoring=scoring_method,
            n_jobs=-1,
            cv=predefined_split_object
        )

        start_time = time()
        random_search.fit(full_X_data, full_y_data)
        elapsed_time = time() - start_time

        # mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)
        # mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # with mlflow.start_run(run_name="CAT_RandomSearch", nested=True):
        #     for param_name, param_value in random_search.best_params_.items():
        #         mlflow.log_param(param_name, param_value)

        #     mlflow.log_metric('Tuning Time (Seconds)', elapsed_time)
        #     mlflow.log_param("dataset_used", "data/creditcard.csv")
        #     mlflow.log_metric('Best Score', random_search.best_score_)
        #     mlflow.set_tag('Model', 'CatBoost')

        return random_search.best_params_

    def voting_ranomized_search(
            self,
            X_train:pd.DataFrame,
            y_train:pd.Series,
    ):

        logger.info(f'Loading Start Of XGB model for Voting (classifier)')

        xg_start = time()
        xg_params = self.xgb_randomized_search(X_train,y_train)
        logger.info(f'XGB Parameter Successfully Tunned Time Takken: {time() - xg_start}')

        xg = XGBClassifier(**xg_params)
        xg.fit(X_train,y_train)
        
        xg_time = time() - xg_start
        logger.info(f'Loading OF XGB model Completed Time Takken: {xg_time}')
        logger.info('Loading Started Of LGB model For Voting (Classifier)')

        lg_start = time()
        lg_params = self.lgb_randomized_search(X_train,y_train)        
        logger.info(f'LGB Paramenter Successfully Tunned Time Takken: {time() - lg_start}')

        lg = LGBMClassifier(**lg_params)
        lg.fit(X_train,y_train)
        lg_time = time() - lg_start
        logger.info(f'Loading OF LGB model Completed Time Takken: {lg_time}')
        cat_start = time()
        logger.info('Loading Started Of CAT model For Voting (Classifier)')
        cat_params = self.catboost_randomized_search(X_train,y_train)
        logger.info(f'CAT Paramenter Successfully Tunned Time Takken: {time() - cat_start}')
        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train,y_train)

        cat_time = time() - cat_start
        logger.info(f'Loading OF CAT model Completed Time Takken: {cat_time}')

        votting = VotingClassifier(
                        estimators=[
                            cast(Tuple[str,BaseEstimator],('XGB',xg)),
                            cast(Tuple[str,BaseEstimator],('LGB',lg)),
                            cast(Tuple[str,BaseEstimator],('CAT',cat))
                        ],
                        voting='soft',
                        n_jobs=-1,
                    )

        return votting

    def xgb_optuna_tune(
            self,
            trial: optuna.trial.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> float:
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.1,
            random_state=self.random_seed_value
        )

        xgb_optuna_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 3.0),
            'random_state': self.random_seed_value,
            'eval_metric': 'logloss',
        }

        model = XGBClassifier(**xgb_optuna_params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )


        preds = model.predict(X_val_split)
        
        score = f1_score(y_val_split, preds)

        return float(score)

    def lgb_optuna_tune(
            self,
            trial: optuna.trial.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> float:
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.1,
            random_state=self.random_seed_value,
            stratify=y_train
        )

        lgb_optuna_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'random_state': self.random_seed_value,
            'n_jobs': -1
        }

        model = LGBMClassifier(**lgb_optuna_params)

        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            callbacks=[
                early_stopping(stopping_rounds=self.early_stopping_rounds),
                log_evaluation(0)
            ]
        )

        preds = model.predict(X_val_split)
        preds = cast(np.ndarray,preds)
        score = f1_score(y_val_split, preds)

        return float(score)

    def cat_optuna_tune(
            self,
            trial: optuna.trial.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> float:
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=0.1,
            random_state=self.random_seed_value
        )

        catboost_optuna_params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'loss_function': 'Logloss',
            'verbose': 0,
            'random_state': self.random_seed_value
        }

        model = CatBoostClassifier(**catboost_optuna_params)

        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            early_stopping_rounds=self.early_stopping_rounds
        )

        preds = model.predict(X_val_split)
        score = f1_score(y_val_split, preds)

        return float(score)
    
    def votting_optuna_tune(
            self,
            X_train:pd.DataFrame,
            y_train:pd.Series
    ):
        logger.info(f'Loading Start Of XGB model for Voting (classifier)')

        xg_start = time()
        
        def xgb_wrap(trial):
            return self.xgb_optuna_tune(trial,X_train,y_train)
        xg_study = optuna.create_study(direction='maximize')
        xg_study.optimize(xgb_wrap,n_trials=30,n_jobs=-1)
        xg_params = xg_study.best_params

        logger.info(f'XGB Parameter Successfully Tunned Time Takken: {time() - xg_start}')

        xg = XGBClassifier(**xg_params)
        xg.fit(X_train,y_train)
        
        xg_time = time() - xg_start
        logger.info(f'Loading OF XGB model Completed Time Takken: {xg_time}')

        logger.info('Loading Started Of LGB model For Voting (Classifier)')

        lg_start = time()

        def lgb_wrap(trial):
            return self.lgb_optuna_tune(trial,X_train,y_train)
        lg_study = optuna.create_study(direction='maximize')
        lg_study.optimize(lgb_wrap,n_trials=30,n_jobs=-1)
        lg_params = lg_study.best_params
        
        logger.info(f'LGB Paramenter Successfully Tunned Time Takken: {time() - lg_start}')

        lg = LGBMClassifier(**lg_params)
        lg.fit(X_train,y_train)

        lg_time = time() - lg_start
        logger.info(f'Loading OF LGB model Completed Time Takken: {lg_time}')

        cat_start = time()
        logger.info('Loading Started Of CAT model For Voting (Classifier)')
        def cat_wrap(trial):
            return self.cat_optuna_tune(trial,X_train,y_train)
        
        cat_study = optuna.create_study(direction='maximize')
        cat_study.optimize(cat_wrap,n_trials=30,n_jobs=-1)

        cat_params = cat_study.best_params        

        logger.info(f'CAT Paramenter Successfully Tunned Time Takken: {time() - cat_start}')

        cat = CatBoostClassifier(**cat_params)
        cat.fit(X_train,y_train)

        cat_time = time() - cat_start
        logger.info(f'Loading OF CAT model Completed Time Takken: {cat_time}')

        votting = VotingClassifier(
                        estimators=[
                            cast(Tuple[str,BaseEstimator],('XGB',xg)),
                            cast(Tuple[str,BaseEstimator],('LGB',lg)),
                            cast(Tuple[str,BaseEstimator],('CAT',cat))
                        ],
                        voting='soft',
                        n_jobs=-1,
                    )

        return votting


def get_settings():
    
    return Settings()