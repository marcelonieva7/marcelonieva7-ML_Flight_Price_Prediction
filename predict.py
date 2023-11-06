from flask import Flask
from flask import request, jsonify

import pickle
import numpy as np

with open('model_rf.bin', 'rb') as f_in:
  model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
  dv = pickle.load(f_in)

app = Flask('credit_scoring')

@app.route('/predict', methods=['POST'])
def predict():
  client = request.get_json()
  X = dv.transform([client])
  pred = model.predict(X)[0]
  price = round(np.expm1(pred), 2)

  return jsonify(price)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=9696)