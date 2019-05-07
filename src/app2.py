from flask import Flask, request, jsonify, send_file
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin

import time 
import os
import base64
import predict_ as pred

app = Flask(__name__)
#cors = CORS(app,resources={r"/*":{"origins":"https://text-summarization.q-appliedai.com"}})

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET','POST'])
def file_data():

    if request.method == 'POST':
        print("in post")
#         data = request.stream.read()
#         filename = data

#        with open("input2.txt", 'wb') as f:
#            f.write(data)

        result = pred.summary_text('input1.txt')

        return jsonify({"result": result})

if __name__ == '__main__':
    app.run(port=9900, host='0.0.0.0')





