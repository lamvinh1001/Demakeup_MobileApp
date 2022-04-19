from flask import Flask, request, jsonify
import base64
import numpy as np
from models import Generator
from utils import extract_face, generate
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from gfpgan import GFPGANer
import time
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# -----------------
generator = Generator(HEIGHT=256, WIDTH=256)
generator.load_weights('weight/Originalc_generator.h5')
print('Loaded Cycle GAN')

face_enhancer = GFPGANer(
    upscale=2.5,
    model_path='weight/GFPGANv1.3.pth')

print('Loaded GFPGAN')

key = ""


@app.route('/', methods=['POST'])
def index():
    global key
    start = time.time()
    image_base64 = str(request.form['base64'])
    image_uri = str(request.form['uri'])
    # if image_uri and image_base64:
    name = image_uri.split('/')[-1]
    key = name
    with open("images/"+key, "wb") as f:
        f.write(base64.b64decode(image_base64))
    print('send take' + str(time.time() - start))
    return ''
    # return jsonify({'status': 'Save Successful!'})
    # else:
    #     return jsonify({'status': 'Save Error!'})


@app.route('/predict', methods=['GET'])
def send_base64():
    global key
    start = time.time()

    path = "images/" + str(key)

    image = Image.open(path)

    face_image = extract_face(image)
    image_generate = generate(face_image, generator)
    pill_im = Image.fromarray((image_generate * 255).astype(np.uint8))

    pill_im.save(path)

    in_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    _, _, output1 = face_enhancer.enhance(
        in_img, has_aligned=False, only_center_face=False, paste_back=True)
    cv2.imwrite(path, output1)
    with open(path, "rb") as image_file:
        data = base64.b64encode(image_file.read())
    print('gen take' + str(time.time() - start))

    return data


if __name__ == "__main__":
    #app.run(host='0.0.0.0', port='8000', debug=True)
    app.run(debug=True)
