import os

labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

os.mkdir("Dataset")

for label in LABELS:
    name = "Dataset/%s" % label
    os.mkdir(name)
