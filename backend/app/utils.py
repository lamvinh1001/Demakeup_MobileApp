from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import cv2
import math
from torch.nn import functional as F


def extract_face(image, detector, required_size=(256, 256)):

    image = np.array(image)
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = image[y1-10:y2+15, x1-10:x2+15]
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


def predict_image(model, image):
    img = tf.image.resize(image, (256, 256))
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img_pre = model(img[tf.newaxis, ...], training=False)
    return img_pre
