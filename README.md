# Credit_Card_Fraud_Detection

An end-to-end Machine Leanring Project To Detect Fruads on Credit Card Transactions using Models like XGBClassifier in This Project We Preprocess the Data From credicard.csv Like Log of Transaction Amount etc

## Problem Statement

Credit Card Frauds Posses a Major Threat to financial Systems. Our Goal is to Find out those Fruad Transactions Using Machine Learning Models Like XGboost

## Dataset

### Credit Card Fraud Detection

**Source** : [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]
**Columns**: Total 30 (Time,Class,V1-V28)
**Size**   : 284,807 Transactions
**Fruad Cases**: 0.17 % (Severly Imbalanced)

## Tools & Libraries

- Python 3.10+
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [CatBoost](https://catboost.ai/)
- [Imbalanced-learn (SMOTEENN)](https://imbalanced-learn.org/)
- [Optuna](https://optuna.org/)
- [MLflow](https://mlflow.org/)
- [Typer](https://typer.tiangolo.com/) for CLI

---

## Metrics Used

**Confusion-Matrix**: For Detailed error analysis
**F1-Score** : Balance-Precision & Recall
**ROC-AUC** : Measure Classifier's Ability to ditinguish between Classes
**PR-Curve**: Precision-Recall Visualization For Imabalance Data

## Model Used

**XGboost** : (`XGBClassifier`)

## Cross-Validation

**StratifiedKFold**:(5-Fold) used for Stable Class Representation

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## How To Run

### 1: Install Requirements

````bash

pip install -r requirements.txt

