import pandas as pd
import pickle

from pathlib import Path
from datetime import timedelta


 
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml" / "model.pkl"

DATA_PATH = BASE_DIR / "data" / "sales.csv"


 
def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


 

def predict_demand(product_name):

    model = load_model()

    data = pd.read_csv(
        DATA_PATH
    )

    
    data.columns = (
        data.columns
        .str.strip()
        .str.replace(
            "\ufeff",
            "",
            regex=False
        )
    )

     
    if "date" not in data.columns:

        raise ValueError(
            "The sales.csv file must contain a 'date' column."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["date"]
    )

  
    product_data = data[
        data["product"] == product_name
    ]

    if product_data.empty:

        return None

     
    latest_date = data["date"].max()

    
    future_date = (
        latest_date +
        timedelta(days=7)
    )

    
    week = int(
        future_date.isocalendar().week
    )

    month = future_date.month

    day = future_date.day

    day_of_week = (
        future_date.dayofweek
    )

  
    price = float(
        product_data["price"].iloc[-1]
    )

    
    features = pd.DataFrame(
        [
            {
                "week": week,
                "month": month,
                "day": day,
                "day_of_week": day_of_week,
                "price": price
            }
        ]
    )

     
    prediction = model.predict(
        features
    )[0]

    prediction = max(
        0,
        round(float(prediction))
    )

    return prediction


 

def get_product_insight(
    product_name,
    current_stock
):

    demand = predict_demand(
        product_name
    )

    if demand is None:

        return None

    
    reorder = max(
        0,
        demand - current_stock
    )

    
    if demand >= 40:

        demand_level = "HIGH"

    elif demand >= 20:

        demand_level = "MEDIUM"

    else:

        demand_level = "LOW"

     
    if reorder > 0:

        recommendation = (
            f"AI recommends ordering "
            f"{reorder} more units."
        )

    else:

        recommendation = (
            "Current stock is sufficient."
        )

    return {

        "demand": demand,

        "demand_level":
            demand_level,

        "reorder":
            reorder,

        "recommendation":
            recommendation

    }