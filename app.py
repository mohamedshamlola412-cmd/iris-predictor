from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    # This line tells Flask to look in the /Templates folder for your UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
        prediction = model.predict(features)
        
        # Send the result back as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Render (your hosting) will tell the app which port to use
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
