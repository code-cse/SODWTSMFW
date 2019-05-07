from flask import Flask, request, jsonify, send_file
import time 
import os
app = Flask(__name__)

import predict_ as pred



@app.route('/', methods=['POST'])
def file_data():
    # f = open('input.txt','w')
    tic = time.time()
    print("1")
    if request.method == 'POST':
        print("in post")
        data = request.stream.read()
        print("2")
        filename = data
        print("----------================")
        print("----------================")
        print("----------================")
        print("----------================")
        print(data)
        with open("input.txt", 'w') as f:
            f.write(data)
        print("----------================")
        print("----------================")
        print("----------================")
        # data.write('input.txt')
        # articles = 'input.txt'
        result = pred.summary_text('/home/ashok/Desktop/NLP/1_demo/src/input.txt')
        # print(result)
        print("3")
        toc = time.time() - tic
        print("time ", toc)
        # return jsonify({"image": str(encoded_string), "pred": pred})
        # os.remove('input.txt')
        return jsonify({"result": result})


if __name__ == '__main__':
	app.run(debug = True)