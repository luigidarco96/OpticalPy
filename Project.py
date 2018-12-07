import cv2
import numpy as np
import sys
import argparse
from skimage.transform import resize
import os

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', default="yolov3.cfg", help='path to yolo config file')
ap.add_argument('-w', '--weights', default="yolov3.weights", help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default="yolov3.txt", help='path to text file containing class names')
ap.add_argument('-s', '--save', default=True, help="save to file")
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#=================================================== Main ======================================================
cam = cv2.VideoCapture(0)
subtractor1 = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=True)
numberOfFrame = 0

#Save video
if args.save:
    writer = None
    (W, H) = (None, None)


while True:
    # Increase the number of frame analyzed
    numberOfFrame = numberOfFrame + 1

    ret, img = cam.read()

    #img = resize(img, (img.shape[0] / 4, img.shape[1] / 4), anti_aliasing=True)

    if ret == False:
        cv2.destroyAllWindows()
        sys.exit(0)


    stencil = np.zeros(img.shape).astype(img.dtype)

    # Save video
    if args.save:
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("VideoDemo1.avi", fourcc, 5, (img.shape[1], img.shape[0]), True)

    imgNoBack = subtractor1.apply(img)

    im2, contours, hierarchy = cv2.findContours(imgNoBack, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        cv2.fillConvexPoly(stencil, cnt, [255, 255, 255])
    result = cv2.bitwise_and(img, stencil)


    image = result

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imshow("Object detection", img)

    if args.save:
        writer.write(img)


    keyboard = cv2.waitKey(5)
    if keyboard == 'q' or keyboard == 27:
        break


cv2.destroyAllWindows()
sys.exit