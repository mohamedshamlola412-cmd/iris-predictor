import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- STEP 1: LOAD THE CORRECT MODEL ---
# This looks for the .pkl file you have in GitHub
MODEL_PATH = 'iris_model.pkl'

try:
    if os.path.exists(MODEL_PATH):
        # We use joblib because .pkl is a Scikit-Learn format
        model = joblib.load(MODEL_PATH)
        print("✅ Scikit-Learn model loaded successfully!")
    else:
        print(f"❌ Error: {MODEL_PATH} not found. Check GitHub!")
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
        # Expecting JSON: {"features": [5.1, 3.5, 1.4, 0.2]}
        features = np.array([data['features']])
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Map the numeric result to the flower name
        species = ['Setosa', 'Versicolor', 'Virginica']
        result = species[int(prediction[0])]
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Fixes the 'No open ports' error
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
