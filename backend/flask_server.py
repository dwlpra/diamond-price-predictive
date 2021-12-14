#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))

@app.route('/', methods=['POST'])
def get_prediction():
    req_json = json.loads(request.data) #read request

    data = req_json
    result = loaded_model.predict(data)
    app.logger.info(result)

    result_float = [i for i in result]
    return jsonify({'output': result_float}) # return model output
    
app.run(debug=True)