import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

content = load_image('./data/content_image.png')
style = load_image('./data/style.png')

output = model(tf.constant(content), tf.constant(style))[0]

cv2.imwrite('./results/output.png', cv2.cvtColor(np.squeeze(output)*255, cv2.COLOR_BGR2RGB))