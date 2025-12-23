from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='Templates')

# Load your model
model = joblib.load('iris_model.pkl')

# This dictionary translates the numbers (0, 1, 2) into names
flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction_num = int(model.predict(features)[0])
        
        # Get the name and image URL
        name = flower_names.get(prediction_num, "Unknown")
        
        return jsonify({
            'prediction': name,
            'prediction_num': prediction_num
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)ray(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
