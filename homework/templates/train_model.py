import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "files" / "input" / "house_data.csv"
model_path = BASE_DIR / "homework" / "house_predictor.pkl"

df = pd.read_csv(data_path, sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)

with open(model_path, "wb") as file:
    pickle.dump(estimator, file)