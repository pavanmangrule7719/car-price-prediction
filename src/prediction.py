def predict_price(model, input_data):
    prediction = model.predict(
        input_data
    )

    return prediction