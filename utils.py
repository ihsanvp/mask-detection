import cv2
import tensorflow as tf
import numpy as np


class Colors:
    # (B, G, R)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

class FaceDetector:
    def __init__(self):
        self.model = cv2.CascadeClassifier("data/models/face_detector/haarcascade_frontalface_default.xml")

    def detect(self, image):
        return self.model.detectMultiScale(image, 1.1, 1)

class MaskDetector:
    SCORE_THRESHOLD = 0.5

    LABELS_PATH = "data/models/mask_detector/labels.txt"
    MODEL_PATH = "data/models/mask_detector/keras_model.h5"

    def __init__(self):
        self.model = tf.keras.models.load_model(MaskDetector.MODEL_PATH, compile=False)

    def detect(self, image):
        processed_data = self.preprocess(image)
        predictions = self.model.predict(processed_data, verbose=0)
        prediction = np.squeeze(predictions)
        index = np.argmax(prediction)
        score = prediction[index]
        
        is_match = False

        if index == 1:
            is_match = True
        else:
            is_match = False

        return is_match, score
        

    def preprocess(self, image):
        processed = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        processed = np.asarray(processed, dtype=np.float32)
        processed = np.expand_dims(processed, axis=0)
        processed = (processed / 127.5) - 1
        return processed
    
    @staticmethod
    def load_labels():
        with open(MaskDetector.LABELS_PATH, "r") as f:
            labels_data = f.readlines()

        labels = list(map(lambda x: x.strip(), labels_data))
        return labels
