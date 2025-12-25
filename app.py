import os
import numpy as np
from flask import Flask, request, jsonify

# --- OPTIMIZATION: Disable GPU to save memory & stop CUDA errors ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

app = Flask(__name__)

# --- OPTIMIZATION: Load model GLOBALLY at startup ---
# Replace 'iris_model.h5' with your actual model filename
MODEL_PATH = 'iris_model.h5'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "Iris Predictor is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        # Expecting input like: {"features": [5.1, 3.5, 1.4, 0.2]}
        features = np.array([data['features']])
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Map to Iris names
        species = ['Setosa', 'Versicolor', 'Virginica']
        result = species[int(predicted_class)]
        
        return jsonify({
            'prediction': result,
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
