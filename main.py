import typer
from src.model_training import train_XGB
from src.config import conf
app = typer.Typer()
@app.command(name='Train',help='Model Types: XGB',rich_help_panel='Main Command')

def run(model:str = typer.Option(...,help='Train Model XGB'),
        fold:int = typer.Option(5,help='Number of folds per cross validation')
        ):
    """
    Train models: XGB   using StratifiedKFold
    """
    typer.echo(f'Starting Training {model} With Folds_{fold}')
    conf.n_split = fold
    if model not in ['xgb']:
        typer.echo('Choose the model from xgb')
        raise typer.Exit(1)
    train_XGB()
    typer.echo(f'Training Completed For {model}')

if __name__ == '__main__':
    app()
