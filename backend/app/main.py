from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="Upload a skin lesion image to get a classification with Grad-CAM explainability.",
    version="1.0.0",
)

# Allow the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["prediction"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Skin Lesion Classifier API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}