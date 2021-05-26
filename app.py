import numpy as np
import os
import cv2
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.layers import ReLU

from flask import Flask, redirect, url_for, request, render_template,request
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename




Upload_original = 'generated images/original/'
Upload_segmented = 'generated images/segmented/'
Upload_overlay = 'generated images/overlay/'
model = load_model('C:/Users/vedant/ML&AI/Project/mod/main_v_3.h5')



app = Flask(__name__, template_folder='template')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(500, 500))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)

    preds = np.array(preds,dtype='float32')

    return preds


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload',methods = ['GET','POST'])
def upload():


    for i in os.listdir(Upload_segmented):
        if i is not None:
            os.remove(Upload_segmented+i)
        else:
            pass

    for i in os.listdir(Upload_original):
        if i is not None:
            os.remove(Upload_original+i)
        else:
            pass

    for i in os.listdir(Upload_overlay):
        if i is not None:
            os.remove(Upload_overlay+i)
        else:
            pass



    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(Upload_original,secure_filename(f.filename)))


        for i in os.listdir(Upload_original):
            simg_name = i


        pred = model_predict(Upload_original+simg_name,model)
        pred = pred[0]*255

        cv2.imwrite(Upload_segmented+'segment.jpg',pred)


        over = cv2.imread(Upload_original+simg_name)
        over = cv2.resize(over,(500,500))
        over = np.array(over,dtype='float32')
        # over = over
        lay = cv2.add(over,pred)
        cv2.imwrite(Upload_overlay+'overlay.jpg', lay)
        og = Upload_original+simg_name
        return render_template('index.html',lay_n=og)


if __name__ == '__main__':
    app.run(debug=True)

