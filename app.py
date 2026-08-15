import os
import gc
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

IMAGE_SIZE = (128, 128)

SOILNET_PATH = "SoilNet_93_86.h5"
DETECTOR_PATH = "Soil_Detector.keras"

# Lazy loading or model loading setup
SoilNet = load_model(SOILNET_PATH)
soil_detector = load_model(DETECTOR_PATH)

SOIL_CLASSES = {
    0: ("Alluvial soil", "Alluvial.html"),
    1: ("Black Soil", "Black.html"),
    2: ("Clay soil", "Clay.html"),
    3: ("Invalid", "index.html"),
    4: ("Red soil", "Red.html"),
}

DETECTOR_CLASSES = {0: "non-soil", 1: "soil"}


def is_soil_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = soil_detector.predict(img_array)[0][0]
    tf.keras.backend.clear_session()
    gc.collect()
    return (DETECTOR_CLASSES[1 if prediction > 0.5 else 0]) == "soil"


def classify_soil_type(image_path, model, threshold=0.75):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    confidence_score = np.max(predictions)
    class_idx = np.argmax(predictions)

    tf.keras.backend.clear_session()
    gc.collect()

    if confidence_score < threshold:
        return "Invalid", "index.html", "Low prediction confidence. Please upload a clearer soil image!"

    pred_name, output_page = SOIL_CLASSES.get(class_idx, ("Unknown", "index.html"))

    if pred_name == "Invalid":
        return "Invalid", "index.html", "Please upload a valid soil image!"

    return pred_name, output_page, None


def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded!")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="Please select a valid image file.")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        if not is_soil_image(file_path):
            cleanup_file(file_path)
            return render_template("index.html", error="Invalid image! Please upload a valid soil image.")

        pred_name, output_page, error_msg = classify_soil_type(file_path, SoilNet, threshold=0.75)

        if error_msg:
            cleanup_file(file_path)
            return render_template("index.html", error=error_msg)

        return render_template(output_page, pred_name=pred_name, image_path=filename)

    except Exception as e:
        cleanup_file(file_path)
        return render_template("index.html", error=f"An error occurred while processing the image: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
