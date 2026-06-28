"""
Loads the trained EfficientNet-B0 model, runs predictions, and generates
Grad-CAM explainability heatmaps.
"""
import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Paths ---
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_focal_loss.pt")
CLASS_INFO_PATH = os.path.join(MODELS_DIR, "class_info.json")

# --- Load class info ---
with open(CLASS_INFO_PATH, "r") as f:
    CLASS_INFO = json.load(f)

CLASS_NAMES = CLASS_INFO["class_names"]
IDX_TO_CLASS = {int(k): v for k, v in CLASS_INFO["idx_to_class"].items()}
LABEL_MAP = CLASS_INFO["label_map"]
IMG_SIZE = CLASS_INFO["img_size"]
NUM_CLASSES = CLASS_INFO["num_classes"]
NORM_MEAN = CLASS_INFO["normalize_mean"]
NORM_STD = CLASS_INFO["normalize_std"]

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model (once, at import time - not per-request) ---
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- Grad-CAM setup ---
target_layers = [model.conv_head]
cam = GradCAM(model=model, target_layers=target_layers)

# --- Same preprocessing as training (eval_transform from the notebook) ---
eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])


def predict_with_explanation(image_bytes: bytes):
    """
    Takes raw image bytes, returns prediction + confidence + base64 Grad-CAM overlay.
    """
    # Decode image bytes -> numpy array (BGR, as cv2 expects)
    nparr = np.frombuffer(image_bytes, np.uint8)
    raw_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    raw_image_resized = cv2.resize(raw_image, (IMG_SIZE, IMG_SIZE))

    # Preprocess for model input
    transformed = eval_transform(image=raw_image_resized)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Run prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()

    # All class probabilities (useful for the frontend to show a breakdown)
    all_probs = {
        CLASS_NAMES[i]: float(probs[i].item())
        for i in range(NUM_CLASSES)
    }

    # Grad-CAM heatmap for the predicted class
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_float = raw_image_resized.astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Encode overlay image as base64 PNG (so it can be sent as JSON to the frontend)
    import base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR))
    cam_base64 = base64.b64encode(buffer).decode("utf-8")

    pred_class_code = IDX_TO_CLASS[pred_idx]
    pred_class_full_name = LABEL_MAP[pred_class_code]

    return {
        "predicted_class": pred_class_code,
        "predicted_class_name": pred_class_full_name,
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
        "gradcam_overlay_base64": cam_base64,
    }