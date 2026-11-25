from pathlib import Path
import json

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

# Existing disease classifier
DISEASE_MODEL_ID = "neghani/plantvillage-mobilenetv2"
DISEASE_MODEL_REVISION = "v1.1"
classifier = None
disease_model_error = None

# Separate lightweight plant-species identification model
PLANT_MODEL_REPO = "cpoisson/plantnet300k-mobilenetv3-small"
plant_model = None
plant_labels = []
plant_model_error = None
plant_transform = None


def load_disease_model():
    global classifier, disease_model_error
    try:
        print("Loading plant disease AI model...")
        classifier = pipeline(
            "image-classification",
            model=DISEASE_MODEL_ID,
            revision=DISEASE_MODEL_REVISION
        )
        disease_model_error = None
        print("Plant disease AI model ready!")
    except Exception as exc:
        disease_model_error = str(exc)
        print("Disease model loading failed:", disease_model_error)


def load_plant_model():
    """Load the 1,081-species PlantNet MobileNetV3-Small classifier."""
    global plant_model, plant_labels, plant_model_error, plant_transform

    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
        from huggingface_hub import hf_hub_download

        print("Loading PlantNet plant-identification model...")
        weights_path = hf_hub_download(
            repo_id=PLANT_MODEL_REPO,
            filename="mobilenetv3_small_v2.pth"
        )
        labels_path = hf_hub_download(
            repo_id=PLANT_MODEL_REPO,
            filename="plantnet300K_species_id_2_name.json"
        )

        with open(labels_path, "r", encoding="utf-8") as file:
            label_data = json.load(file)

        if isinstance(label_data, dict):
            keys = list(label_data.keys())
            if all(str(key).isdigit() for key in keys):
                keys = sorted(keys, key=lambda value: int(value))
            else:
                keys = sorted(keys)
            labels = [str(label_data[key]) for key in keys]
        elif isinstance(label_data, list):
            labels = [str(value) for value in label_data]
        else:
            raise ValueError("Unexpected PlantNet species-label file format.")

        model = models.mobilenet_v3_small(weights=None, num_classes=1081)
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if len(labels) != 1081:
            raise ValueError(
                f"Expected 1081 species labels, found {len(labels)}."
            )

        plant_model = model
        plant_labels = labels
        plant_transform = transform
        plant_model_error = None
        print("PlantNet identification model ready (1081 species).")

    except Exception as exc:
        plant_model_error = str(exc)
        print("Plant identification model loading failed:", plant_model_error)


def identify_plant(image):
    if plant_model is None:
        return {
            "available": False,
            "message": "Plant identification model is unavailable.",
            "candidates": []
        }

    try:
        import torch

        tensor = plant_transform(image).unsqueeze(0)
        with torch.inference_mode():
            logits = plant_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            scores, indices = torch.topk(probabilities, k=5)

        candidates = []
        for score, index in zip(scores.tolist(), indices.tolist()):
            candidates.append({
                "name": plant_labels[index],
                "score": round(float(score) * 100, 2)
            })

        top_score = candidates[0]["score"] if candidates else 0
        is_uncertain = top_score < 20

        return {
            "available": True,
            "status": "uncertain" if is_uncertain else "prediction_available",
            "name": candidates[0]["name"] if candidates and not is_uncertain else None,
            "top_score": top_score,
            "candidates": candidates,
            "message": (
                "Plant identity is uncertain. Try a clear photo showing the leaf."
                if is_uncertain else
                "Species prediction generated. Compare the candidate list before relying on it."
            )
        }
    except Exception as exc:
        app.logger.exception("Plant identification failed")
        return {
            "available": False,
            "message": "Plant identification could not be completed.",
            "error": str(exc),
            "candidates": []
        }


def load_models():
    load_disease_model()
    load_plant_model()


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "app": "CropDoctor AI",
        "status": "running",
        "ai_model": "ready" if classifier else "unavailable",
        "plant_identification_model": "ready" if plant_model else "unavailable",
        "disease_model_error": disease_model_error,
        "plant_model_error": plant_model_error
    })


@app.post("/api/scan")
def scan():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a plant image."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        image = Image.open(uploaded_file.stream)
        if image.format not in ALLOWED_FORMATS:
            return jsonify({
                "error": "Use a JPG, PNG or WEBP image."
            }), 400

        image = image.convert("RGB")
        width, height = image.size
        if width < 32 or height < 32:
            return jsonify({"error": "Image resolution is too small."}), 400

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "Invalid image file."}), 400

    # Plant identity and disease classification are independent.
    plant_result = identify_plant(image)
    disease_results = []
    disease_message = "Disease classifier is unavailable."

    if classifier is not None:
        try:
            predictions = classifier(image, top_k=3)
            for item in predictions:
                raw_label = item["label"]
                confidence = float(item["score"])

                if "___" in raw_label:
                    crop, disease = raw_label.split("___", 1)
                else:
                    crop, disease = "Unknown", raw_label

                disease_results.append({
                    "label": raw_label,
                    "crop": crop.replace("_", " "),
                    "disease": disease.replace("_", " "),
                    "confidence": round(confidence * 100, 2)
                })
            disease_message = "Disease prediction generated; verify before taking action."
        except Exception as exc:
            app.logger.exception("Disease prediction failed")
            disease_message = "Disease prediction could not be completed."
            disease_model_error = str(exc)

    if not plant_result.get("available") and not disease_results:
        return jsonify({
            "error": "Both AI models are unavailable. Check the Flask terminal.",
            "plant_model_error": plant_model_error,
            "disease_model_error": disease_model_error
        }), 503

    return jsonify({
        "success": True,
        "width": width,
        "height": height,
        "plant_identification": plant_result,
        "predictions": disease_results,
        "disease_message": disease_message,
        "disease_model": DISEASE_MODEL_ID,
        "disease_model_revision": DISEASE_MODEL_REVISION,
        "plant_model": PLANT_MODEL_REPO,
        "note": (
            "AI outputs are model predictions, not confirmed identifications or diagnoses. "
            "The PlantNet model was evaluated on its own test dataset; results can differ "
            "on local garden and field photos."
        )
    })


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    load_models()
    app.run(debug=False)
