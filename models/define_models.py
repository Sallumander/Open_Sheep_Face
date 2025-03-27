import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
import joblib
from ultralytics import YOLO
import numpy as np




class EfficientNetV2RegionDetector(nn.Module):
    def __init__(self, num_regions=5):  
        super(EfficientNetV2RegionDetector, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, num_regions * 4),
            nn.Sigmoid()  # Ensures all predictions are in [0,1]
        )

    def forward(self, x):
        return self.backbone(x)

# Define the landmark detection model
class LandmarkNet(nn.Module):
    def __init__(self, num_landmarks):
        super(LandmarkNet, self).__init__()
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, num_landmarks * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

def get_models(device):
   
    region_detector = EfficientNetV2RegionDetector().to(device)
    checkpoint = torch.load("models/region/region_detector_transform.pth", map_location=device)

    # Rename incorrect keys
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("backbone.classifier.1.0.0.", "backbone.classifier.1.0.")  # Fix classifier key mismatch
        new_state_dict[new_key] = v

    # Load modified state_dict
    region_detector.load_state_dict(new_state_dict)
    region_detector.eval()

    # Load the trained landmark detection models
    landmark_models = {
        'left_ear': LandmarkNet(num_landmarks=4).to(device),
        'eyes': LandmarkNet(num_landmarks=4).to(device),
        'right_ear': LandmarkNet(num_landmarks=4).to(device),
        'nose_mouth': LandmarkNet(num_landmarks=10).to(device),
        'forehead': LandmarkNet(num_landmarks=3).to(device)
    }

    MODEL_PATHS = [
        "left_ear_landmark_detector_regions.pth",
        "eyes_landmark_detector_regions.pth",
        "right_ear_landmark_detector_regions.pth",
        "nose_mouth_landmark_detector_regions.pth",
        "forehead_landmark_detector_regions.pth"
    ]

    for i, (name, model) in enumerate(landmark_models.items()):
        model.load_state_dict(torch.load("models/landmarks/"+MODEL_PATHS[i], map_location=device))
        model.eval()

    return region_detector, landmark_models
    
class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._init_models()
        return cls._instance

    def _init_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[ModelLoader] Loading YOLO model...")
        self.yolo_model = YOLO("models/face/best.pt")

        print("[ModelLoader] Loading region + landmark models...")
        self.region_detector, self.landmark_models = get_models(self.device)

        # Optional: convert to half precision for speed (only on CUDA)
        if self.device.type == "cuda":
            self.region_detector.half()
            for model in self.landmark_models.values():
                model.half()

    def get_yolo(self):
        return self.yolo_model

    def get_region_detector(self):
        return self.region_detector

    def get_landmark_models(self):
        return self.landmark_models

    def get_pain_model(self):
        return self.pain_model

    def get_device(self):
        return self.device
