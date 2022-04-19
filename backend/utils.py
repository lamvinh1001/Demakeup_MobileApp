# import cv2
import matplotlib.pyplot as plt
from models import Generator
from PIL import Image
import numpy as np
import cv2

# bg_upsampler=upsampler
# from tensorflow.keras.models import load_model
# srgan = load_model('weight/srgan.h5')


# generator = Generator(HEIGHT=256, WIDTH=256)
# generator.load_weights('weight/Originalc_generator.h5')
# print('Loaded')


def extract_face(image, required_size=(256, 256)):
    from mtcnn import MTCNN
    detector = MTCNN()

    image = np.array(image)
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def generate(face_array, generate):

    image = (face_array - 127.5) / 127.5
    output = generate.predict(image.reshape(
        1, 256, 256, 3)).reshape(256, 256, 3)
    output = (output + 1) / 2

    return output

    # return output1
    # output = cv2.normalize(output, None, alpha=0, beta=255,
    #    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # conver to base64
    # base64_code = base64.b64encode(output)

    # return base64_code


# if __name__ == "__main__":

#     image = cv2.imread('1.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # _, _, output1 = face_enhancer.enhance(
#     #     image, has_aligned=False, only_center_face=False, paste_back=True)
#     # cv2.imshow('sz', output1)
#     # cv2.waitKey()
#     face_img = extract_face(image)
#     # # face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

#     generate(face_img, generator)

    # cv2.imshow('sz', demakeup_img)
    # cv2.waitKey()

    # print(srgan_image.shape)
    # cv2.imwrite('rs.png', (srgan_image[0] * 100).astype(np.uint8))
