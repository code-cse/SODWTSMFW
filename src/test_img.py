import execnet

from flask import Flask, request, jsonify, send_file
import time 
import os
import base64
app = Flask(__name__)

# import nmf as pred

# import predict_ as pred
import os 

@app.route('/', methods=['GET','POST'])

def read():

    file_name = "/home/ashok/Desktop/NLP/1_demo/src/imgs/nmf_topic_top_words.jpg"

    with open(file_name, "rb") as image_file:
        val = base64.b64encode(image_file.read())
    # val1 = "Hello"
    return jsonify({"result": val})

if __name__ == '__main__':
    app.run(port=9990, host='0.0.0.0', debug = True)