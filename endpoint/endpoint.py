import base64
import re
from io import BytesIO
import uuid
import cv2
import numpy
import json
import os
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from functions.model import process

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def index():
    return 'Hello'


@app.route('/upload', methods=['POST'])
def upload():
    data = json.loads(request.data)
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    id = '/tmp/%s.%s' % (uuid.uuid1(), im.format)

    im.save(id)
    npimg = numpy.fromfile(id, numpy.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    os.remove(id)
    return process(img)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
