import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload directory
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Target image size for models
IMAGE_SIZE = (128, 128)

# Load Models at application startup
SOILNET_PATH = "SoilNet_93_86.h5"
DETECTOR_PATH = "Soil_Detector.keras"

SoilNet = load_model(SOILNET_PATH)
soil_detector = load_model(DETECTOR_PATH)

# SoilNet Multi-Class Mapping: {index: (class_name, template_name)}
SOIL_CLASSES = {
    0: ("Alluvial soil", "Alluvial.html"),
    1: ("Black Soil", "Black.html"),
    2: ("Clay soil", "Clay.html"),
    3: ("Invalid", "index.html"),
    4: ("Red soil", "Red.html"),
}

# Binary Detector Class Mapping (0: non-soil, 1: soil)
DETECTOR_CLASSES = {0: "non-soil", 1: "soil"}


def is_soil_image(image_path):
    """Binary classifier check to verify if the uploaded image is soil."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = soil_detector.predict(img_array)[0][0]
    predicted_label = DETECTOR_CLASSES[1 if prediction > 0.5 else 0]

    return predicted_label == "soil"


def classify_soil_type(image_path, model, threshold=0.75):
    """Multi-class classifier to predict specific soil type."""
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    confidence_score = np.max(predictions)
    class_idx = np.argmax(predictions)

    # Reject low-confidence classifications
    if confidence_score < threshold:
        return "Invalid", "index.html", "Low prediction confidence. Please upload a clearer soil image!"

    pred_name, output_page = SOIL_CLASSES.get(class_idx, ("Unknown", "index.html"))

    if pred_name == "Invalid":
        return "Invalid", "index.html", "Please upload a valid soil image!"

    return pred_name, output_page, None


def cleanup_file(file_path):
    """Utility function to safely delete uploaded files on error."""
    if os.path.exists(file_path):
        os.remove(file_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded!")

    file = request.files["file"]
    if file.filename == "":
        return render_template(
            "index.html", error="Please select a valid image file."
        )

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Step 1: Run Binary Soil Detector
        if not is_soil_image(file_path):
            cleanup_file(file_path)
            return render_template(
                "index.html", error="Invalid image! Please upload a valid soil image."
            )

        # Step 2: Run Multi-Class Soil Classifier
        pred_name, output_page, error_msg = classify_soil_type(
            file_path, SoilNet, threshold=0.75
        )

        if error_msg:
            cleanup_file(file_path)
            return render_template("index.html", error=error_msg)

        # Render output page with saved image path
        return render_template(
            output_page, pred_name=pred_name, image_path=filename
        )

    except Exception as e:
        cleanup_file(file_path)
        return render_template(
            "index.html", error=f"An error occurred while processing the image: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)