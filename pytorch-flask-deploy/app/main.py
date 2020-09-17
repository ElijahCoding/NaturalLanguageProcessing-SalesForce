from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # load image
    # image -> tensor
    # prediction
    # return json
    return jsonify({
        'result': 1
    })