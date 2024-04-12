import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov9TestConstants:
    YOLOV7_C_MODEL_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt"
    YOLOV7_C_MODEL_PATH = "C:/Users/Green Sturgeon/AI_Project/TrainYoloV9/yolov9/yolov9-c.pt

    YOLOV9_E_MODEL_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt"
    YOLOV9_E_MODEL_PATH = "C:/Users/Green Sturgeon/AI_Project/TrainYoloV9/yolov9/yolov9-e.pt


def download_yolov7c_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9_C_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9_C_MODEL_URL,
            destination_path,
        )


def download_yolov9e_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov9TestConstants.YOLOV9_E_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov9TestConstants.YOLOV9_E_MODEL_URL,
            destination_path,
        )
