# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("trainData.csv")

X = df[
    [
        "Eye_Blinks",
        "Head_Movements",
        "Hand_Movements",
        "Body_Movements"
    ]
]

y = df["FIS"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

pickle.dump(
    model,
    open("anxiety_model.pkl", "wb")
)

print("Model trained and saved!")