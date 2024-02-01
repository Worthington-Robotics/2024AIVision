import cv2
import argparse
import numpy as np
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

from blob import processFrame

CLASSES = yaml_load(check_yaml("./models/data.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

BOUNDING_BOX_EXPAND = 20


def main(onnx_model, input_image):
    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    
    # Load the ONNX Model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1/255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)
         ) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply Non-Maximum suppression
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale
        }
        detections.append(detection)
        drawBoundingBox(original_image, class_ids[index], scores[index], round(
            box[0] * scale), round(box[1] * scale), round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
    cv2.imwrite("image.jpg", original_image)

    return detections


def process(model: cv2.dnn.Net, original_image: np.ndarray):
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1/255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    # Perform inference
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)
         ) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply Non-Maximum suppression
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    detections = []
    new_image = original_image.copy()
    mask = np.zeros(original_image.shape[:2], dtype="uint8")
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale
        }
        detections.append(detection)
        drawBoundingBox(new_image, class_ids[index], scores[index], round(box[0] * scale), round(
            box[1] * scale), round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        print(class_ids[index])
        if CLASSES[class_ids[index]] == 'note':
            cv2.rectangle(mask, (round(box[0] * scale) - BOUNDING_BOX_EXPAND, round(box[1] * scale) - BOUNDING_BOX_EXPAND), (round(
                (box[0] + box[2]) * scale) + BOUNDING_BOX_EXPAND, round((box[1] + box[3]) * scale) + BOUNDING_BOX_EXPAND), 255, -1)
    masked = cv2.bitwise_and(original_image, original_image, mask=mask)
    processFrame(masked)
    # cv2.imshow("image.jpg", masked)
    # cv2.waitKey(1)

    return detections


def drawBoundingBox(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
    # './models/FRC2024.onnx'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/FRC2024.onnx",
                        help="Inputs the filename of your ONNX model.")
    parser.add_argument("--img", default="bus.jpg", help="Path to image")
    parser.add_argument("--mode", default="img",
                        help="Mode to run in, img or live")
    args = parser.parse_args()
    if args.mode == "img":
        main(args.model, args.img)
    elif args.mode == "live":
        cam = cv2.VideoCapture(0)
        # Load the ONNX Model
        model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(args.model)
        while True:
            success, frame = cam.read()
            process(model, frame)
    else:
        print("Invalid mode")
