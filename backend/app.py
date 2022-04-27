from flask import Flask, request, jsonify, session
import base64
import numpy as np
from models import Generator
from utils import extract_face, generate
from io import BytesIO
from PIL import Image
from gfpgan import GFPGANer
import os
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.secret_key = 'my-kltn'
# -----------------

generator = Generator(HEIGHT=256, WIDTH=256)
generator.load_weights('weight/Originalc_generator.h5')
print('Loaded Cycle GAN')

face_enhancer = GFPGANer(
    upscale=2.5,
    model_path='weight/GFPGANv1.3.pth')

print('Loaded GFPGAN')


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
    import time
    if "image_name" in session:
       # load image
        img_name = session["image_name"]
        path = "images/" + str(img_name)
        original_img = Image.open(path)

        # extract face
        face_img = extract_face(original_img)
        os.remove(path)
        # using CycleGAN to generate de-makeup face thực thi ~2.3s ở local
        img_generate = generate(face_img, generator)
        pill_imgG = Image.fromarray((img_generate * 255).astype(np.uint8))

        # using gfpgan to enhance face thực thi ~ 3.5s ở local
        _, _, img_enhance = face_enhancer.enhance(np.array(
            pill_imgG)[:, :, ::-1], has_aligned=False, only_center_face=False, paste_back=True)
        pill_imgE = Image.fromarray(img_enhance[:, :, [2, 1, 0]])

        # conver to base64
        buffered = BytesIO()
        pill_imgE.save(buffered, format="JPEG")
        result = base64.b64encode(buffered.getvalue())
        return result
    else:
        return jsonify({'status': 'Gen Error!'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8000', threaded=True)
    # app.run(threaded=True)
