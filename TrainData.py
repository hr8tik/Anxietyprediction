# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

df = pd.read_csv("trainData.csv")

# Select only available features (Hand_Movements may have been removed)
# Prefer training without hand feature (useful if hand detection removed)
preferred = ["Eye_Blinks", "Head_Movements", "Body_Movements"]
candidate_features = ["Eye_Blinks", "Head_Movements", "Hand_Movements", "Body_Movements"]

available = {c for c in df.columns}

# Use preferred set if available; otherwise fall back to any available candidate features
if all(f in available for f in preferred):
    available_features = preferred
else:
    available_features = [c for c in candidate_features if c in available]

if not available_features:
    raise ValueError("No training features available in trainData.csv")

X = df[available_features]
y = df["FIS"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model next to this script
model_path = os.path.join(os.path.dirname(__file__), "anxiety_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {model_path} with features: {available_features}")