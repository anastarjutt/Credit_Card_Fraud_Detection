import typer
import numpy as np

from src.training.model_training import get_model_training
from src.config.paths import get_path
from src.config.settings import get_settings

train = get_model_training()
settings = get_settings()
paths = get_path()

app = typer.Typer()
app.command(name='Train',help='Train a Model: [XGB, LGB, CAT]',rich_help_panel='Main Command')

def run(
        model:str = typer.Option(...,help='Specify A Model: [XGB, LGB, CAT]'),
        fold:int = typer.Option(5,help='Number of Folds For Cross-Validation')
):
    """
    Train A Model Using Cross-Validation (StratifiedKFold) for Fraud Detection.
    """

    model = model.upper()

    if model not in ['XGB','LGB','CAT']:
        typer.secho('Invalid Model Name Choose One: XGB LGB or CAT',fg=typer.colors.RED)
        typer.Exit(1)
    
    typer.echo(f'Trainin {model} with {fold} Folds')
    settings.n_split = fold

    try:
        if model == 'XGB':
            all_f1,all_roc,all_cm,all_clf,all_results = train.train_xgb()
        elif model == 'LGB':
            all_f1,all_roc,all_cm,all_clf,all_results = train.train_lgb()
        elif model == 'CAT':
            all_f1,all_roc,all_cm,all_clf,all_results = train.train_cat()
        
        typer.secho(f'Training Compeleted Successfully for {model} Model.',fg=typer.colors.GREEN)
        typer.echo(f'Avg F1_Score: {np.mean(all_f1):.2f}')
        typer.echo(f'Avg Roc_Auc Score: {np.mean(all_roc):.2f}')

    except Exception as e:
        typer.secho(f'Training Failed for {model} Model Due to error: {e}',fg=typer.colors.RED)
        raise typer.Exit(1)
if __name__ == '__main__':
    app()