import cv2
import numpy as np

# Функция отрисовки рамки и названия обнаруженного объекта


def draw_object(img, index, box):
    with open('coco.names.txt') as f:
        classes = f.read().split('\n')   # Получаем названия классов объектов
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    img = cv2.rectangle(img, start, end, color, width)   # Отрисовка рамки вокруг объекта

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    img = cv2.putText(img, text, start, font, font_size, color, width, cv2.LINE_AA)  # Отрисовка названия обнаруженного объекта

    return img

# Функция обнаружения объектов на изображении


def detect_image(image):

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    img = cv2.imread(image)
    height, width, depth = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (608,608), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes]
    outs = net.forward(out_layers)

    class_indexes = []
    class_scores = []
    boxes = []

    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                x = center_x - obj_width // 2
                y = center_y - obj_height // 2

                box = [x, y, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index[0]
        img = draw_object(img, class_indexes[box_index], boxes[box_index])

    cv2.imwrite('dog_output.jpg', img)
