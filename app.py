from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load model and scaler once
MODEL_PATH = Path("model/best_wine_model.pkl")
SCALER_PATH = Path("model/scaler.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Home route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        alcohol = float(request.form["alcohol"])
        malic_acid = float(request.form["malic_acid"])
        ash = float(request.form["ash"])
        alcalinity_of_ash = float(request.form["alcalinity_of_ash"])
        magnesium = float(request.form["magnesium"])
        total_phenols = float(request.form["total_phenols"])
        flavanoids = float(request.form["flavanoids"])
        nonflavanoid_phenols = float(request.form["nonflavanoid_phenols"])
        proanthocyanins = float(request.form["proanthocyanins"])
        color_intensity = float(request.form["color_intensity"])
        hue = float(request.form["hue"])
        od280_od315_of_diluted_wines = float(request.form["od280_od315_of_diluted_wines"])
        proline = float(request.form["proline"])

        features = np.array([[ 
            alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
            total_phenols, flavanoids, nonflavanoid_phenols,
            proanthocyanins, color_intensity, hue,
            od280_od315_of_diluted_wines, proline
        ]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_class = int(prediction[0])

        return render_template("index.html", prediction=predicted_class)

    except:
        return render_template("index.html", error="Prediction failed. Check inputs.")

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
