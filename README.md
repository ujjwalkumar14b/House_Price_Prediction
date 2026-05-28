# House_Price_Prediction
![Deployment Screenshot](deployment.png)

## Overview
This project implements a machine learning-based house price prediction system. It analyzes housing data and predicts the price of a house based on their features like area and location.

The solution includes:
* End-to-end data preprocessing pipeline
* Feature engineering
* Model training and evaluation
* Web deployment using Flask and Bootstrap

## Dataset
The dataset used is: `train.csv` and `test.csv`

It contains housing-related features such as:
* Locations (Neighborhood)
* Year (YearBuilt, YearRemodAdd, GarageYrBlt, YrSold)
* Condition (Condition1, Condition2, OverallCond, ExterCond, BsmtCond)
* Quality ( OverallQual, ExterQual, BsmtQual, KitchenQual)
* Target variable `SalePrice`

## Project Structure

```
House_Price_Prediction/
│
├── app.py                          # Flask application
├── House_Price_Prediction.ipynb    # Model training notebook
├── House_Price_Prediction.csv      # Dataset
├── House_Price_Prediction.pkl      # Trained model
├── templates/
│   └── index.html                 # Frontend UI
├── deployment.png                 # App preview
├── requirements.txt               # Requirements
├── setup.py                       # Package Installation
└── README.md
```

## Machine Learning Pipeline

### 1. Data Preprocessing
* Missing value handling using `SimpleImputer`
* Feature scaling using `StandardScaler`
* One-hot encoding for categorical variables

### 2. Models Used
* Linear Regression
* Random Forest Regressor

### 3. Evaluation Metrics
* MAE 
* MSE 
* RMSE
* R2 Score

## Model Selection
The final model is selected based on **R2 score**: The model with the higher R2_Score is chosen automatically

## Web Application Features
* User-friendly form using Bootstrap
* Radio button inputs for binary features (Yes/No)
* Real-time price prediction

## Installation
```
git clone https://github.com/ujjwalkumar14b/House_Price_Prediction.git
cd House_Price_Prediction
pip install -r requirements.txt
python app.py
```

## Deployment
The application can be deployed on:
* Render
* Railway
* AWS EC2
* Heroku (if configured)

## Key Learnings
* End-to-end ML pipeline design
* Handling real-world data inconsistencies
* Model serialization using joblib
* Resolving pickle dependency issues
* Building and deploying ML web apps

## Future Improvements
* Add REST API support
* Model monitoring and logging
* Hyperparameter tuning
* Use advanced models (XGBoost, LightGBM)
* Frontend enhancement (React)

## Author
Ujjwal Kumar
GitHub: [https://github.com/ujjwalkumar14b](https://github.com/ujjwalkumar14b)

## License
This project is open-source and available under the MIT License.
