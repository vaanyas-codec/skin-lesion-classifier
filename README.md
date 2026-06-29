# Skin Lesion Classifier

A deep learning system that classifies dermatoscopic skin lesion images into 7 diagnostic categories, with a focus on minimizing missed melanoma cases. Built end-to-end: model training, explainability analysis, REST API, and a deployed web app.

**Live demo:** https://skin-lesion-classifier-seven.vercel.app
**API:** https://skin-lesion-classifier-production.up.railway.app/docs

> This is a research/portfolio project trained on a public dataset. It is not a diagnostic tool and should not be used for medical decisions.

---

## The problem

Classifying skin lesions from images is a 7-class problem (melanoma, basal cell carcinoma, melanocytic nevi, and others) with severe class imbalance — benign moles outnumber dangerous lesions by as much as 58:1 in the training data. A model optimized for raw accuracy will happily ignore the rare, dangerous classes, since it can hit 90%+ accuracy just by always predicting "benign mole."

The real objective isn't accuracy — it's **minimizing missed melanomas**, since a false negative here is far costlier than a false positive.

## Model

- **Architecture:** EfficientNet-B0, transfer learning (ImageNet-pretrained, fine-tuned)
- **Dataset:** HAM10000, 10,015 dermatoscopic images, 7 classes
- **Framework:** PyTorch, trained on Google Colab (free GPU tier)

### Two training iterations - a deliberate tradeoff

| | Loss function | Overall accuracy | Melanoma recall |
|---|---|---|---|
| v1 | Weighted Cross-Entropy | 75.1% | 59.9% |
| v2 | Focal Loss (gamma=2.0) | 72.2% | 71.9% |

The first version optimized for overall accuracy and looked better on paper, but it missed 4 out of every 10 actual melanoma cases. Switching to focal loss traded 3 points of overall accuracy for a 12-point improvement in melanoma recall.

### Explainability (Grad-CAM)

Grad-CAM heatmaps confirmed the model consistently localizes attention on the lesion itself, even on misclassified examples. See the training notebook for full results.

## Architecture & deployment

- **Backend:** FastAPI, single /api/predict endpoint, returns predicted class, confidence, and full probability breakdown
- **Frontend:** React + Vite, drag-and-drop upload, live probability visualization
- **Deployment:** Railway (backend), Vercel (frontend)

Grad-CAM is not included in the live API - pytorch_grad_cam has a hard internal dependency on OpenCV/libGL not reliably available on minimal cloud containers. It remains in the training notebook with full results.

## Known limitations

- Melanoma recall is ~72%, not 100% - this is a research demo, not a clinical tool
- Out-of-distribution inputs (non-dermatoscope photos) produce unreliable, low-confidence predictions
- Dataset bias: HAM10000 skews toward certain skin tones and clinical sources

## Tech stack

PyTorch, timm, Grad-CAM, FastAPI, React, Vite, Railway, Vercel
