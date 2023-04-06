import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['OPENCV_VIDEOIO_CAPTURE_BACKEND'] = 'dshow'

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# cv2.dnn.cuda.setDevice(0)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# print(layer_names)
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Resize and normalize the image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Feed the image to the network
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                label = classes[class_id]
                print(label)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
