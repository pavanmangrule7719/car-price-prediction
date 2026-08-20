import pandas as pd

from src.model_saving import load_model
from src.prediction import predict_price


model = load_model(
    "models/final_model.pkl"
)


car = pd.DataFrame([
    {
        "model": " A1",
        "year": 2019,
        "mileage": 25000,
        "transmission": "Manual",
        "fuelType": "Petrol",
        "tax": 150,
        "mpg": 55.4,
        "engineSize": 1.0
    }
])


prediction = predict_price(
    model,
    car
)


print(
    f"Predicted Price: ${prediction[0]:.2f}"
)