import os

def create_folder():

    if not os.path.isdir("Dataset"):

        labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")

        os.mkdir("Dataset")

        for label in LABELS:
            name = "Dataset/%s" % label
            os.mkdir(name)


if __name__ == "__main__":
    create_folder()
