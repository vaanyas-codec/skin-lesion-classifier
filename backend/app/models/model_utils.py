"""
Loads the trained EfficientNet-B0 model and runs predictions.
Note: Grad-CAM explainability is documented in the training notebook (Colab) with
real example outputs - it is intentionally NOT included in this deployed API,
since pytorch_grad_cam has a hard internal dependency on OpenCV/cv2, which requires
a system-level libGL library not reliably available on minimal cloud containers
(Render/Railway free tiers). See the project README/notebook for Grad-CAM results.
"""
import json
import os
from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import timm

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

# --- Normalization constants as numpy arrays (reused per-request) ---
_NORM_MEAN_ARR = np.array(NORM_MEAN, dtype=np.float32)
_NORM_STD_ARR = np.array(NORM_STD, dtype=np.float32)


def eval_transform(image: np.ndarray):
    """
    Manual replacement for albumentations' eval_transform (resize + normalize + to-tensor).
    No cv2/albumentations dependency - pure numpy/torch.
    `image` is expected to already be resized to IMG_SIZE x IMG_SIZE, RGB, uint8 HWC.
    """
    img_float = image.astype(np.float32) / 255.0
    img_normalized = (img_float - _NORM_MEAN_ARR) / _NORM_STD_ARR
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    return {"image": img_tensor}


def predict_with_explanation(image_bytes: bytes):
    """
    Takes raw image bytes, returns prediction + confidence + full probability breakdown.
    (No Grad-CAM overlay in the deployed API - see module docstring for why.)
    """
    # Decode image bytes -> PIL Image -> numpy array (RGB)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_image_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    raw_image_resized = np.array(pil_image_resized)

    # Preprocess for model input
    transformed = eval_transform(raw_image_resized)
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

    pred_class_code = IDX_TO_CLASS[pred_idx]
    pred_class_full_name = LABEL_MAP[pred_class_code]

    return {
        "predicted_class": pred_class_code,
        "predicted_class_name": pred_class_full_name,
        "confidence": round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
    }