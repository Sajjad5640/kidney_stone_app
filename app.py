from flask import Flask, render_template, request
import os
from utils.detector import detect_and_crop
from utils.classifier import classify_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run YOLO detection + get crops
    detection_result, crops = detect_and_crop(filepath, RESULT_FOLDER)

    final_predictions = []

    for organ_name, crop_list in crops.items():
        for crop_info in crop_list:     # crop_info is a dict
            crop_path = crop_info["crop"]   # or "upscaled"

            pred = classify_image(crop_path, organ_name)

            final_predictions.append({
                "organ": organ_name,
                "prediction": pred,
                "crop": crop_path,
                "upscaled": crop_info["upscaled"],
                "bbox": crop_info["bbox"]
            })


    return render_template(
        "result.html",
        original_image=filepath,
        detection_result=detection_result,
        predictions=final_predictions
    )

# -------------------------------------------------------------
# 🔥 XAI ROUTE → This is where the Explain button will go
# -------------------------------------------------------------
@app.route("/xai", methods=["POST"])
def xai():
    crop_path = request.form["crop_path"]
    organ = request.form["organ"]

    # generate XAI
    pred_label, xai_image = classify_image(
        crop_path, organ, generate_xai_flag=True
    )

    return render_template(
        "xai_result.html",
        crop=crop_path,
        organ=organ,
        xai_image=xai_image,
        prediction=pred_label
    )
if __name__ == "__main__":
    app.run(debug=True)