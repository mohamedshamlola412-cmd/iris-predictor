from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# Ensuring we use your 'Templates' folder
app = Flask(__name__, template_folder='Templates')

# Load your model
model = joblib.load('iris_model.pkl')

# Dictionary to map numbers to names
flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        
        # Get the name from our dictionary
        name = flower_names.get(prediction, "Unknown")
        
        return jsonify({'prediction': name})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
