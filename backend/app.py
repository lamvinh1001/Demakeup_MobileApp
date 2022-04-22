from flask import Flask, request, jsonify, session
import base64
import numpy as np
from models import Generator
from utils import extract_face, generate

from PIL import Image
import cv2
from gfpgan import GFPGANer
import time
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.secret_key = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcba'
# -----------------

face_enhancer = GFPGANer(
    upscale=2.5,
    model_path='weight/GFPGANv1.3.pth')

print('Loaded GFPGAN')

generator = Generator(HEIGHT=256, WIDTH=256)
generator.load_weights('weight/Originalc_generator.h5')
print('Loaded Cycle GAN')


@app.route('/')
def index():
    return 'template null'


@app.route('/post', methods=['POST'])
def save_base64():
    # global key
    # start = time.time()
    if request.method == 'POST':
        # print('join')
        image_base64 = str(request.form['base64'])
        image_uri = str(request.form['uri'])
        # if image_uri and image_base64:
        name = image_uri.split('/')[-1]

        session['image_name'] = name
        with open("images/"+session['image_name'], "wb") as f:
            f.write(base64.b64decode(image_base64))
        # print('send take' + str(time.time() - start))

        return jsonify({'status': 'Save Successful!'})

    return jsonify({'status': 'Save Error!'})


@app.route('/predict', methods=['GET'])
def send_base64():
    # global key
    # start = time.time()
    if "image_name" in session:

        # start = time.time()
        image_name = session["image_name"]
        path = "images/" + str(image_name)

        image = Image.open(path)

        face_image = extract_face(image)
        image_generate = generate(face_image, generator)

        pill_im = Image.fromarray((image_generate * 255).astype(np.uint8))

        pill_im.save(path)
        # print(time.time() - start)

        in_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        #start1 = time.time()

        _, _, output1 = face_enhancer.enhance(
            in_img, has_aligned=False, only_center_face=False, paste_back=True)
        # print(output1.shape)
        cv2.imwrite(path, output1)
        with open(path, "rb") as image_file:
            data = base64.b64encode(image_file.read())
        # print('gen take' + str(time.time() - start))
        # print(time.time() - start1)
        return data
    else:
        return jsonify({'status': 'Gen Error!'})


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port='8000', debug=True, threaded=True)
    app.run(threaded=True)
