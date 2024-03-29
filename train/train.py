from ultralytics import YOLO
from roboflow import Roboflow
import argparse
import os

def main(key):
    model = YOLO('yolov8n.pt')

    rf = Roboflow(api_key=key)
    project = rf.workspace("worbots-4145").project("2024-frc")
    version = project.version(8)
    dataset = version.download("yolov8")

    model.train(data=f"{dataset.location}/data.yaml", epochs=900, imgsz=640, patience=900, plots=True)
    model.val()
    model.export(format='onnx')
    # version.deploy("yolov8", "./")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="", help="Your roboflow API key")
    args = parser.parse_args()
    main(args.key)