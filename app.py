from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

print("Loading model...")
model = joblib.load("heart_model.pkl")
print("Model loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        # Insert dataset column (0 = Cleveland default)
        features.insert(1, 0)

        features = np.array(features, dtype=float).reshape(1, -1)
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "High Risk of Heart Disease"
        else:
            result = "Low Risk of Heart Disease"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
