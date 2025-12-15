import pickle
import numpy as np
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f:
    model = pickle.load(f)

app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = np.array([[client[k] for k in sorted(client.keys())]])
    proba = model.predict_proba(X)[0, 1]

    return jsonify({
        'conversion_probability': float(proba),
        'conversion': proba >= 0.5
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9696)