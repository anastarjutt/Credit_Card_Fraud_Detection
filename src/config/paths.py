import os
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','.env'))
load_dotenv(dotenv_path=env_path)

class paths:
    def __init__(self) -> None:
        
        self.BASE_DIR = os.path.abspath(os.path.join(os.getcwd(),'..','..'))
        
        self.NOTEBOOKS_PATH = os.path.join(self.BASE_DIR,'notebooks')


        self.DATA_PATH = os.getenv('DATA_PATH','data')
        self.DATA_CSV_PATH = os.path.join(self.DATA_PATH,'creditcard.csv') 

        self.MODELS_PATH = os.getenv('MODEL_PATH',os.path.join(self.BASE_DIR,'models'))
        self.XGB_PATH = os.path.join(self.MODELS_PATH,'xgb')
        self.LGB_PATH = os.path.join(self.MODELS_PATH,'lgb')
        self.FOLD_PATH = os.path.join(self.MODELS_PATH,'fold')
        self.CAT_PATH = os.path.join(self.MODELS_PATH,'cat')
        self.VOTNG_PATH = os.path.join(self.MODELS_PATH,'voting')

        self.REPORT_PATH = os.getenv('REPORT_PATH','reports')
        self.LOG_PATH = os.getenv('LOG_PATH','logs')
        self.LOG_FILE = os.getenv('LOG_FILE')

def get_path():
    return paths()