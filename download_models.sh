#!/usr/bin/env bash

echo "Downloading config files..."

mkdir yolo-coco
wget -O yolo-coco/coco.data https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/coco.data
wget -O yolo-coco/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget -O yolo-coco/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -O yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
python create_directory.py
