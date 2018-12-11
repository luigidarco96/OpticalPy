import cv2
import sys
import argparse
import os
import logging
import time
from bgsub import bgsub
from yolo import yolo
import numpy as np
import uuid

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    hf, wf = flow.shape[:2]
    yf, xf = np.mgrid[step/2:hf:step, step/2:wf:step].reshape(2,-1).astype(int)
    fx, fy = flow[yf,xf].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    return vis

def create_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(name+'_log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


# create a logger
log = create_logger("opticalPy")
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to input video")
ap.add_argument("-s", "--save",
                help="path to save video")
ap.add_argument("-y", "--yolo",
                default="yolo-coco",
                help="base path to YOLO directory")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
log.info("open coco class at " + args["yolo"])
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

log.info("open yolo weights at " + args["yolo"])
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
log.info("open yolo config at " + args["yolo"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

log.info("loading YOLO from disk")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Check if processing video or take frame from webcam
if args.get("video", None) is None:
    log.info("loading webcam")
    cam = cv2.VideoCapture(0)
    # wait a bit for camera sensor warming up
    time.sleep(2.0)
else:
    log.info("loading "+args["video"])
    cam = cv2.VideoCapture(args["video"])


if not cam.isOpened:
    log.error("An error occurred when trying to open the video")
    exit(0)

# Init the background subtract
bgimg = bgsub(cv2.bgsegm.createBackgroundSubtractorGMG(1, 0.9))

# Save video
if args["save"]:
    log.info("Start recording video "+args["save"])
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = None


yolo = yolo(net, LABELS)

_, first = cam.read()
prvs = cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)

numberOfFrame = 0
while True:
    numberOfFrame = numberOfFrame + 1
    ret, img = cam.read()
    if ret == False:
        cv2.destroyAllWindows()
        sys.exit(0)

    bgsubimg = bgimg.get_bgsub_frame(img)
    outs = yolo.create_blob(bgsubimg)
    
    next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    Width = img.shape[1]
    Height = img.shape[0]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        yolo.draw_prediction(img, class_ids[i], confidences[i], round(
            x), round(y), round(x + w), round(y + h))
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if x < 0:
            x = 0
        if x + w > img.shape[0]:
            w = img.shape[0] - x
        if y < 0:
            y = 0
        if y + h > img.shape[1]:
            h = img.shape[1] - y
        flow_cut = flow[y:(y+h),[range(x,(x+w))]][:,0,:,:]
        name_flow = uuid.uuid4()
        name_directory = yolo.get_label(class_ids[i])
        save_path = "Dataset/%s/%s" % (name_directory, name_flow)
        np.save(save_path, flow_cut)
        

    cv2.imshow("OpticalPy", img)
    #cv2.imshow("bgsub", bgsubimg)

    # Save video
    if args["save"]:
        if writer is None:
            writer = cv2.VideoWriter(
                args["save"]+".avi", fourcc, 5, (img.shape[1], img.shape[0]), True)
        writer.write(img)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cam.release()
cv2.destroyAllWindows()
