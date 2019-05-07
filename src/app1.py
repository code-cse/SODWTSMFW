from flask import Flask, request, jsonify, send_file, render_template, session, redirect, url_for
from flask_cors import CORS, cross_origin
import nmf as pred
import base64
import time 
import os
import requests
import json


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
def file_data():
    if request.method == 'POST':
        print("in post")
        data = request.stream.read()
        filename = data

        with open("input1.txt", 'wb') as f:
            f.write(data)

        result = pred.main('input1.txt',1)

        image = str(read())
        app2 = requests.post('http://0.0.0.0:9900/').content
        app2 = app2.decode().replace('\n', "")
        app2 = json.loads(app2)
        app2_result = app2['result']
        return jsonify({"result": result,
                        "image" : image,
                        "app2_result": app2_result})

def read():

    file_name = "imgs/top_words.jpg"

    with open(file_name, "rb") as image_file:
        val = base64.b64encode(image_file.read())
    return val


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(port=5004, host='0.0.0.0', debug=True)
    # app.run()
