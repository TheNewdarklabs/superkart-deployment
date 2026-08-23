# Flask API that serves the serialized SuperKart sales-forecast pipeline.
# The serialized artifact `superkart_model.joblib` is a full sklearn Pipeline
# (preprocessing + regressor), so this API accepts raw human-readable input.

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask("SuperKart Sales Predictor")

# Load once at startup so every request reuses the same in-memory model.
model = joblib.load("superkart_model.joblib")

FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_MRP", "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Store_Age_Years", "Product_Type_Category",
]


@app.get("/")
def home():
    return "SuperKart Sales Prediction API is running. See /v1/predict and /v1/predictbatch."


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/v1/predict")
def predict_sales():
    """Online (single-record) inference. Body: JSON with the 10 features."""
    payload = request.get_json(force=True)
    input_df = pd.DataFrame([payload])[FEATURES]
    prediction = float(model.predict(input_df)[0])
    return jsonify({"Predicted sales": round(prediction, 2)})


@app.post("/v1/predictbatch")
def predict_batch():
    """Batch inference. Form-data: `file` = CSV with the 10 feature columns."""
    file = request.files["file"]
    input_df = pd.read_csv(file)[FEATURES]
    predictions = model.predict(input_df)
    return pd.Series(predictions).astype(float).round(2).to_json(double_precision=2)


if __name__ == "__main__":
    # host 0.0.0.0 is required for the port to be reachable from outside the container.
    app.run(host="0.0.0.0", port=7860, debug=False)
