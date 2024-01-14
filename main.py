import cv2
import argparse

model: cv2.dnn.Net = cv2.dnn.readNetFromONNX('./models/FRC2024.onnx')
outputs = model.forward()