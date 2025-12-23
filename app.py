from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# FIXED: We merged lines 6 and 7 into one single command
app = Flask(__name__, template_folder='Templates')

# Load your model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    # This will now find index.html inside your "Templates" folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
