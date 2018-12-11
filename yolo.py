import numpy as np
import cv2


class yolo:
    def __init__(self, net, labels):
        self.net = net
        layer_names = self.net.getLayerNames()
        self.layer_output = [layer_names[i[0] - 1]
                             for i in self.net.getUnconnectedOutLayers()]
        self.labels = labels
        # take some random color
        np.random.seed(42)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.labels), 3))

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.labels[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def create_blob(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.layer_output)

    def get_label(self, class_id):
        return self.labels[class_id]
