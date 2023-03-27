from flask import Flask, render_template, request, jsonify
import numpy as np

import os

from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

from keras import models
from keras import layers
from keras.models import Sequential
import keras.utils as image
from tensorflow import keras
app = Flask(__name__)
# model = load_model('model.h5')
# model.make_predict_function()


def make_global_model():
    print('You pressed the button!')
    k = os.listdir("./models")
    print(k)
    model_grp = []
    for i in k:
        m = load_model("./models/"+i)
        model_grp.append(m)

    weights = []
    for i in model_grp:
        weight = i.get_weights()
        weights.append(weight)

    l = len(weights)
    new_weights = [sum(x) for x in zip(*weights)]
    new_weights = [x/l for x in new_weights]
    model_grp[0].set_weights(new_weights)
    model_grp[0].save("./models/global_model.h5")
    print("Ceated the global model....")
# routes


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Flask app....."


@app.route("/globalmodel", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        try:

            make_global_model()

        # return label
            return render_template("index.html", result="Done")
        except Exception as e:
            return e
    else:
        return "Error"


@app.route("/predictclass", methods=['GET', 'POST'])
def predict_class():
    model = models.load_model("./models/global_model.h5")

    if request.method == 'POST':
        img = request.files['my_image']
        image_path = './static/images/'+img.filename
        img.save(image_path)

        img = image.load_img(image_path, target_size=(32, 32))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x, axis=0)
        y_pred_task1, _ = model.predict(x)
        class_names = ['airplane', 'automobile', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = class_names[np.argmax(y_pred_task1)]
        return render_template("index.html", prediction=label, img_path=image_path)


@app.route("/predictclass2", methods=['GET', 'POST'])
def predict_class2():
    model = models.load_model("./models/global_model.h5")

    if request.method == 'POST':
        img = request.files['my_image']
        image_path = './static/images/'+img.filename
        img.save(image_path)
        img = image.load_img(image_path, target_size=(32, 32))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x, axis=0)
        nul, y_pred_task2 = model.predict(x)
        label = '{}'.format('It is a ship ' if y_pred_task2 >
                            0.5 else 'It is not a ship')
        return render_template("index.html", prediction=label, img_path=image_path)


if __name__ == '__main__':

    app.run(debug=True)
