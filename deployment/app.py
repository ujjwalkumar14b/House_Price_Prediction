from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

MODEL_PATH = "../models/final_house_price_model.pkl"
model = joblib.load(MODEL_PATH)

# =====================================================
# USER-EDITABLE FEATURES (ONLY THESE SHOW IN FORM)
# =====================================================
NUM_FEATURES = [
    "LotFrontage","LotArea","YearBuilt","YearRemodAdd",
    "GrLivArea","FullBath","BedroomAbvGr",
    "GarageCars","GarageArea","MoSold","YrSold"
]

CAT_FEATURES = [
    "MSZoning","Neighborhood","HouseStyle",
    "GarageType","SaleType","SaleCondition"
]

# =====================================================
# DROPDOWN OPTIONS
# =====================================================
OPTIONS = {
    "MSZoning": ["RL","RM","FV","RH","C (all)"],
    "Neighborhood": ["NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt"],
    "HouseStyle": ["1Story","2Story","1.5Fin","SLvl"],
    "GarageType": ["Attchd","Detchd","BuiltIn","None"],
    "SaleType": ["WD","New","COD"],
    "SaleCondition": ["Normal","Partial","Abnorml"]
}

# =====================================================
# 🔴 ALL TRAINING FEATURES WITH DEFAULT VALUES
# (THIS FIXES YOUR ERROR)
# =====================================================
DEFAULTS = {
    # ID
    "Id": 0,

    # Numerical
    "LotFrontage": 70, "LotArea": 9500,
    "OverallQual": 5, "OverallCond": 5,
    "YearBuilt": 1975, "YearRemodAdd": 1995,
    "MasVnrArea": 0,
    "BsmtFinSF1": 400, "BsmtFinSF2": 0,
    "BsmtUnfSF": 500, "TotalBsmtSF": 900,
    "1stFlrSF": 1100, "2ndFlrSF": 0,
    "LowQualFinSF": 0, "GrLivArea": 1500,
    "BsmtFullBath": 1, "BsmtHalfBath": 0,
    "FullBath": 2, "HalfBath": 0,
    "BedroomAbvGr": 3, "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 6, "Fireplaces": 1,
    "GarageYrBlt": 1975, "GarageCars": 2,
    "GarageArea": 480,
    "WoodDeckSF": 0, "OpenPorchSF": 40,
    "EnclosedPorch": 0, "3SsnPorch": 0,
    "ScreenPorch": 0, "PoolArea": 0,
    "MiscVal": 0, "MoSold": 6, "YrSold": 2010,

    # Engineered
    "HouseAge": 35, "RemodAge": 15,
    "TotalSF": 2000, "TotalBath": 2.5,

    # Categorical
    "MSSubClass": "20",
    "MSZoning": "RL",
    "Street": "Pave",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinType2": "Unf",
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "KitchenQual": "TA",
    "Functional": "Typ",
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageFinish": "Unf",
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "SaleType": "WD",
    "SaleCondition": "Normal"
}

# =====================================================
# ROUTE
# =====================================================
@app.route("/", methods=["GET","POST"])
def index():
    prediction = None

    if request.method == "POST":

        # 1️⃣ Start with ALL defaults (CRITICAL)
        data = DEFAULTS.copy()

        # 2️⃣ Override ONLY user inputs
        for col in NUM_FEATURES:
            data[col] = float(request.form[col])

        for col in CAT_FEATURES:
            data[col] = request.form[col]

        # 3️⃣ Create DataFrame with ALL columns
        df = pd.DataFrame([data])

        # 4️⃣ Predict
        prediction = np.expm1(model.predict(df)[0])

    return render_template(
        "index.html",
        num_features=NUM_FEATURES,
        cat_features=CAT_FEATURES,
        options=OPTIONS,
        defaults=DEFAULTS,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
