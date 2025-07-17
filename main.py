import typer
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from typing import cast,Literal
from src.training.model_training import get_model_training,tuner_typer
from src.config.paths import get_path
from src.config.settings import get_settings

train = get_model_training()
paths = get_path()
settings = get_settings()

app = typer.Typer()

@app.command(name='Train',help='Train One of these Models [XGB,LGB,Voting]',rich_help_panel='Main Command')
def run(
    model:str = typer.Option(...,help='Specify One Model From [XGB,LGB,Voting]'),
    fold:int = typer.Option(5,help='Numbers Of Iterations For Cross-Validations'),
    tuner:str = typer.Option('randomized',help='Specify The Model Tunning Method: [optuna,randomized]')
):
    model = model.upper()
    tuner = tuner.lower()

    if model not in ['XGB','LGB','CAT','VOTING']:
        typer.secho("Invalid Model Typer Select From 'XGB','LGB','CAT','VOTING'",fg=typer.colors.RED)
        raise typer.Exit(1)
    
    if tuner not in ['optuna','randomized']:
        typer.secho("Invalid Tunning Method Name Choose from 'optuna','randomized'")
        raise typer.Exit(1)
    
    typer.echo(f'Starting The Training of {model} Model with {fold} Folds')
    settings.num_splits = fold

    settings.tuning_method = cast(Literal['optuna','randomized'],tuner)
    tuner_casted = cast(tuner_typer,tuner)

    if model == 'XGB':
        all_f1,all_roc,all_cm,all_clf,all_results = train.train_xgb(tuner_casted)
    elif model == 'LGB':
        all_f1,all_roc,all_cm,all_clf,all_results = train.train_lgb(tuner_casted)
    elif model == 'CAT':
        all_f1,all_roc,all_cm,all_clf,all_results = train.train_cat(tuner_casted)
    elif model == 'VOTING':
        all_f1,all_roc,all_cm,all_clf,all_results = train.train_voting(tuner_casted)
    
    typer.secho(f'✅ Training Completed for {model}.', fg=typer.colors.GREEN)
    typer.echo(f'Avg F1 Score: {np.mean(all_f1):.4f}')
    typer.echo(f'Avg ROC AUC: {np.mean(all_roc):.4f}')

@app.command(name='Predict',help='Make Predictions Using Saved Models',rich_help_panel='Main Command')
def predict(
    model_path:str = typer.Option(...,help='Path To Saved Model (.pkl)'),
    input_csv: str = typer.Option(...,help='Path To Input Csv File'),
    output_csv: str = typer.Option('Predictions.csv',help='Path To Save New Predictions')
):
    typer.echo(f'Loading Model From Path: {model_path}')
    model = joblib.load(model_path)

    typer.echo(f'Reading Input Data From: {input_csv}')
    data = pd.read_csv(input_csv)

    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])
    
    preds = model.predict(data)

    out_df = pd.DataFrame({'Predictions':preds})

    typer.echo(f'Saving Predictions To New CSV: {output_csv}')
    out_df.to_csv(output_csv,index=False)

    typer.secho('Prediction Completed Successfully',fg=typer.colors.RED)

if __name__ == '__main__':
    app()