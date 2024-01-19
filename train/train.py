from ultralytics import YOLO
from roboflow import Roboflow
import os

model = YOLO('yolov8n.pt')

rf = Roboflow(api_key="fi7NJBusg0KIhDQaFFlK")
project = rf.workspace("raleigh-slack-idx6f").project("2024-frc")
dataset = project.version(5).download("yolov8")

model.train(data=f"{dataset.location}/data.yaml", epochs=500, imgsz=640)
model.val()
model.export(format='onnx')
