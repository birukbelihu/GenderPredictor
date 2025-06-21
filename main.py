import time
from collections import deque

import cv2
import numpy
import numpy as np
from tensorflow.keras.models import load_model

from constants import *

previous_frame_time = 0
fps_history = deque(maxlen=10)

face_net = cv2.dnn.readNetFromCaffe(get_face_detector_prototext_file(), get_face_detector_caffe_model())
face_mask_model = load_model(get_gender_predictor_model())

video_capture = cv2.VideoCapture(1)

while video_capture.isOpened():
    is_successful, frame = video_capture.read()
    if not is_successful:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - previous_frame_time + 1e-5)
    previous_frame_time = new_frame_time

    fps_history.append(fps)
    average_fps = sum(fps_history) / len(fps_history)
    fps_text = f"FPS: {int(average_fps)}"

    cv2.putText(frame, fps_text, (10, 25), cv2.QT_FONT_NORMAL,
                0.7, (219, 109, 24), 2)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                resized_face = cv2.resize(face, (224, 224))
                face_array = resized_face / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                prediction = face_mask_model.predict(face_array, verbose=0)
                prediction_index = numpy.argmax(prediction[0])
                confidence = prediction[0][prediction_index]
                label = get_gender_predictor_model_classes()[numpy.argmax(prediction[0])]
                label_text = f"{label}: ({confidence * 100:.1f}%)"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.QT_FONT_NORMAL,
                            0.7, (0, 255, 0), 2)

    cv2.imshow(get_app_name(), frame)

    key = cv2.waitKey(1) & 0xFF
    if chr(key) in exit_keys():
        print(f"Exiting {get_app_name()}...")
        break

video_capture.release()
cv2.destroyAllWindows()
