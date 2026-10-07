import pandas as pd
import pickle
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


 
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sales.csv"

MODEL_PATH = BASE_DIR / "ml" / "model.pkl"

METRICS_PATH = BASE_DIR / "ml" / "model_metrics.json"


 
print("====================================")
print("       SHOP SYNC AI - ML MODEL")
print("====================================")


 
print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)

 
data.columns = (
    data.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print("\nDataset loaded successfully!")

print("Rows:", len(data))

print("Columns:", list(data.columns))


 
required_columns = [
    "date",
    "product",
    "category",
    "units_sold",
    "price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in sales.csv: {missing_columns}"
    )


 
data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)

data = data.dropna(
    subset=["date"]
)


 
data["week"] = (
    data["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

data["month"] = data["date"].dt.month

data["day"] = data["date"].dt.day

data["day_of_week"] = (
    data["date"]
    .dt.dayofweek
)


 
X = data[
    [
        "week",
        "month",
        "day",
        "day_of_week",
        "price"
    ]
]

y = data["units_sold"]

 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


 
print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


 
predictions = model.predict(
    X_test
)

 
mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


 
with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

 
metrics = {

    "model": "Random Forest Regression",

    "mae": round(
        float(mae),
        2
    ),

    "r2_score": round(
        float(r2),
        2
    ),

    "training_records": len(X_train),

    "testing_records": len(X_test)

}


with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


 
print("\n====================================")
print("       MODEL TRAINING COMPLETED")
print("====================================")

print(
    "Model:",
    metrics["model"]
)

print(
    "Mean Absolute Error:",
    metrics["mae"]
)

print(
    "R2 Score:",
    metrics["r2_score"]
)

print(
    "Training Records:",
    metrics["training_records"]
)

print(
    "Testing Records:",
    metrics["testing_records"]
)

print("\nAI model saved successfully!")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Metrics:",
    METRICS_PATH
)

print("\n====================================")
print("          TRAINING FINISHED")
print("====================================")