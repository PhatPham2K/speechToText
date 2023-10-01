import cv2
import time
import imutils
import numpy as np
import os

# https://github.com/aakashjhawar/face-detection
# Load the cascade
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_model)


def faceDetection():
    # To capture video from webcam.
    cap = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

    while True:
        # Read the frame
        _, img = cap.read()

        frame = imutils.resize(img, width=1600, height=900)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
            # you can also change the 'confidence' (0.5 here) for better results
            if confidence < 0.5:
                continue

            # compute the (x,y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame

        window_name = "Face detection system"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, frame)
        # Close windows with Esc
        key = cv2.waitKey(1) & 0xFF

        # break the loop if ESC key is pressed
        if key == 27:
            break

        #
        #
        # # Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # Detect the faces
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # # Draw the rectangle around each face
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # # Display
        # cv2.imshow('img', img)
        # # Stop if escape key is pressed
        # k = cv2.waitKey(30) & 0xff
        # if k==27:
        #     break
        # Release the VideoCapture object
    cap.release()

# faceDetection()