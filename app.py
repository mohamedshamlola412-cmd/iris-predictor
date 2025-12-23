from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load your model - Ensure this filename matches exactly what is in your GitHub
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    # This renders the index.html file from your /Templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Converts input data to the correct format for the model
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Uses the port provided by Render or defaults to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
        
        # Send the result back as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Render (your hosting) will tell the app which port to use
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
