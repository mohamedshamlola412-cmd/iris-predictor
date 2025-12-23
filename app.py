
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# This loads the 'brain' file you just saved
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return "Iris Model API is Live and Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the numbers from the user request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Send the result back as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Render (your hosting) will tell the app which port to use
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
