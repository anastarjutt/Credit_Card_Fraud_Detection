import os
from dotenv import load_dotenv

load_dotenv()
class config:
    n_split = int(os.getenv('N_SPLIT',5))
    random_seed = int(os.getenv('RANDOM_SEED',42))
    def __init__(self) -> None:
        try:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
        except NameError:
            self.base_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))

        
        self.models_path = os.getenv('DATA_PATH','data/creditcard.csv')
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        self.logs_path = os.getenv('LOG_PATH')
        self.voting_path = os.path.join(self.base_dir,'models','voting')
        self.optuna_path = os.path.join(self.base_dir,'models','optuna')
        self.fold_path = os.path.join(self.base_dir,'models','fold')
        self.cat_path = os.path.join(self.base_dir,'models','cat')
        self.xgb_path = os.path.join(self.base_dir,'models','xgb')
        self.notebooks_path = os.path.join(self.base_dir,'notebooks')
        self.reports_path = os.path.join(self.base_dir,'reports')
        self.data_path = os.path.join(self.base_dir,'data','creditcard.csv')
        self.src_path = os.path.join(self.base_dir,'src')
    
        self.params = {
            'n_estimators':[100,500],
            'max_depth':[3,9],
            'learning_rate':[0.01,0.2],
            'subsample':[1.0,10.0],
            'colsample_bytree':[1.0,10.0],
            'reg_alpha':[0.0,1.0],
            'reg_lambda':[0.0,1.0],
            'scale_pos_weight':[0,255]
    }
    
    def randomized_tune(self,params,X_train,y_train):
        from sklearn.model_selection import RandomizedSearchCV
        from xgboost import XGBClassifier

        xg = XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=conf.random_seed
            )
        
        random_search = RandomizedSearchCV(
            estimator=xg,param_distributions=params,
            cv=conf.n_split,n_jobs=-1,
            scoring='f1',verbose=1
        )
        random_search.fit(X_train,y_train)
        best_params = random_search.best_params_
        return best_params
conf = config()