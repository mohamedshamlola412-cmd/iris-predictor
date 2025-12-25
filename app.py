import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras

# 1. Initialize Flask app
app = Flask(__name__)

# 2. Load the model safely
# We load it globally so it stays in memory between requests
MODEL_PATH = 'model.h5' # Ensure this file is in your GitHub repo!

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            # Using tensorflow's internal keras engine
            model = keras.models.load_model(MODEL_PATH)
            print("--- Model Loaded Successfully ---")
            return model
        except Exception as e:
            print(f"--- Error loading model file: {e} ---")
            return None
    else:
        print(f"--- Model file {MODEL_PATH} not found! ---")
        return None

model = load_model()

@app.route('/')
def home():
    return "Iris Predictor API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        # Expecting JSON input like: {"features": [5.1, 3.5, 1.4, 0.2]}
        data = request.get_json()
        features = np.array([data['features']])
        
        prediction = model.predict(features)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        
        # Mapping (Standard Iris classes)
        classes = ['Setosa', 'Versicolor', 'Virginica']
        result = classes[predicted_class]

        return jsonify({
            'prediction': result,
            'class_index': predicted_class,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Render uses the 'PORT' environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
