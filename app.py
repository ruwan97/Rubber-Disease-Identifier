from crypt import methods
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO

from flask import Flask,request,render_template,jsonify

app = Flask(__name__)

#Load the model
model = load_model('model/rubber_leaf_disease_detection_custom_model.h5')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('predictbycapture', methods=['POST'])
def predictbycapture():

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    
    # image = Image.open(r'img_20220605_123607.jpg')
    # convert captured 64based img to image
    image = Image.open( BytesIO(base64.b64decode(request.form['test_img'].split(',')[1])) )

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)

    # Lables
    # 0 Colletotrichum
    # 1 Corynespora
    # 2 Oidium
    # 3 BG

    colletotrichum = round(prediction[0][0] * 100,2)
    corynespora = round(prediction[0][1] * 100,2)
    oidium = round(prediction[0][2] * 100,2)

    if (colletotrichum > oidium) and (colletotrichum > corynespora):
        max_val = colletotrichum
        max_clone = "Colletotrichum"
    elif (oidium > colletotrichum) and (oidium > corynespora):
        max_val = oidium
        max_clone = "Oidium"
    else:
        max_val = corynespora
        max_clone = "Corynespora"

    # validation is there have any classes are equals
    if(colletotrichum==corynespora) or (colletotrichum==oidium):
        valid = False
    elif (corynespora == colletotrichum) or (corynespora == oidium):
        valid = False
    elif (oidium == colletotrichum) or (oidium == corynespora):
        valid = False
    else:
        valid = True

    # validate max result has at least 51 confident level, If not it will be BG class
    if (max_val<51):
        valid = False
    else:
        valid = True

    return jsonify(colletotrichum=colletotrichum,oidium=oidium,corynespora=corynespora,max_val=max_val,max_clone=max_clone,valid=valid)


if __name__ == '__main__':
    app.run()

