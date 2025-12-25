import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- MODEL LOADING ---
# This matches your file: iris_model.pkl
MODEL_PATH = 'iris_model.pkl'

try:
    if os.path.exists(MODEL_PATH):
        # joblib is used for .pkl files (Scikit-Learn)
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: {MODEL_PATH} not found.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    status = "Online" if model else "Online but MODEL MISSING"
    return f"Iris Predictor is {status}!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file missing on server'}), 500
    
    try:
        data = request.get_json()
        # Expecting a list of 4 floats for the Iris features
        features = np.array([data['features']])
        
        # Make prediction using the Scikit-Learn model
        prediction = model.predict(features)
        
        # Typical Iris class mapping
        species = ['Setosa', 'Versicolor', 'Virginica']
        result = species[int(prediction[0])]
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Uses the port Render provides to avoid connection errors
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
