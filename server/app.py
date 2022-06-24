# -*- coding: utf-8 -*-
from flask import Flask, request, Response, jsonify
from flask import render_template
import pickle
import torch
import model.simple_cnn as simple_cnn
import torchvision.transforms as transforms
import io,os
from PIL import Image
import cv2
import torch.nn as nn
import numpy as np
import base64
from model.chatbot import loadChatbotModel, processUserInput


app = Flask(__name__, static_url_path='')
app.models = {}

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/image', methods=['POST'])
def receiveImage():
    data = request.get_json()
    print(data)
    app.receivedData = data
    return "ok"

@app.route('/lesson1/lr_predict', methods=['POST'])
def lr_predict():
    json = request.get_json()
    print(json)
    values = json["data"]["user_values"]
    X = []
    for v in values:
        X.append([v])

    model = app.models.get("lesson1_1", None)
    if not model:
        print("missing model lesson1_1")
    
    result = model.predict(X).tolist()
    resp = jsonify({"result": result})
    
    return resp

@app.route('/lesson2/mnist_predict', methods=['POST'])
def mnist_predict():
    # check if the post request has the file part
    json = request.get_json()
    userData = json["data"]
    userData = userData.replace("data:image/jpeg;base64,", "")
    imgData = base64.b64decode(userData)

    tempFilename = "mnist_test.jpg"
    f = open(tempFilename, "wb")
    f. write(imgData)
    f. close()

    result = predict(tempFilename)

    return jsonify({"result": result})


@app.route('/lesson3/chatbot', methods=['POST'])
def chatbot():
    # check if the post request has the file part
    json = request.get_json()
    userData = json["data"]["user_input"]
    print(userData)
    response = processUserInput(userData)

    return jsonify({"result": response})



def load_models():
    modelFilename = './data/lesson1_1.p'
    loaded_model = pickle.load(open(modelFilename, 'rb'))
    app.models["lesson1_1"] = loaded_model

    model = simple_cnn.Model().to("cpu")
    model.load_state_dict(torch.load("./data/mnist_cnn.pt"))
    model.eval()
    app.models["lesson2"] = model

    loadChatbotModel()



def predict(filename):
    print(filename)

    tensor = transform_image(filename)
    tensor=tensor.reshape([1,1,28,28])

    model = app.models["lesson2"]
    outputs = model.forward(tensor)
    outputs = outputs.tolist()[0]

    predict_result = np.asarray(outputs).argmax().astype(np.int32)
    result = int("{0}".format(predict_result))
    print("prediction: {}".format(result))
    return result

def transform_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])


    image_tensor = transform(image)

    return image_tensor



if __name__ == '__main__':
    app.debug = True

    load_models()
    print(predict("./data/num1.jpg"))

    app.run(host='0.0.0.0', threaded=True)
