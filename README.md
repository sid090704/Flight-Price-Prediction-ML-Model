# Flight Price Analysis & Prediction

This repository contains a complete end-to-end machine learning project that analyzes and predicts flight ticket prices using multiple modeling approaches: linear regression (with target transformation), Random Forest, and XGBoost. The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, and model saving for deployment.

## Repository Structure

```
Flight_Price_Analysis_and_Prediction/
│
├── notebooks/
│   ├── eda_and_fe.ipynb                # EDA and Feature engineering of datasset
│   ├── flight_prediction_models.ipynb
│   
│
├── data/
│   └── flight_price_eda.csv   #Cleaned dataset in csv 
    ├── cleaned_flight_price.xlsx   #cleaned dataet in excel
│   ├── flight_price.xlsx           #original dataset in excel
│      
│
├── models/
│   ├── lr_model.pkl  #linear regression model
│   ├── rf_model.pkl   #random forest model
│   ├── xgb_model.joblib #xgboost model (best)
│
├── src/
│   ├── preprocessing.py            # Build preprocessing pipeline
│   ├── train_utils.py              # Training helpers (splits, tuning)
│   └── evaluation.py               # Evaluation helpers and metrics
│
├── README.md
├── requirements.txt

```

## Key Results 
- Linear regression (TTR / sqrt target): `R² ≈ 0.68`, `RMSE ≈ 2577`
- Random Forest (tuned): `R² ≈ 0.89`, `RMSE ≈ 1485`
- XGBoost (tuned): `R² ≈ 0.91`, `RMSE ≈ 1350`
  `MAPE: 7.53%`,
  `Model Accuracy: 92.47%` for the XGBoost model

## How to run

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      
pip install -r requirements.txt
```

2. Open the notebooks to reproduce analysis and models:

```bash
jupyter notebook notebooks/flight_prediction_models.ipynb
```

3. Trained models are saved in `models/`. Use the notebooks or `src/` functions to load and predict.



