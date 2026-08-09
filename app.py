import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload directory
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
model_path = "SoilNet_93_86.h5"
SoilNet = load_model(model_path)

# Class Mapping based on exact training directory order
classes = {
    0: ("Alluvial soil", "Alluvial.html"),
    1: ("Black Soil", "Black.html"),
    2: ("Clay soil", "Clay.html"),
    3: ("Invalid", "index.html"),
    4: ("Red soil", "Red.html"),
}


def model_predict(image_path, model, threshold=0.75):
    # Preprocess Image
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict Class Probabilities
    prediction = model.predict(image)[0]
    
    # Get highest probability score and predicted class index
    confidence_score = np.max(prediction)
    result = np.argmax(prediction)

    # If confidence is less than 75%, reject the image as Non-soil / Invalid
    if confidence_score < threshold:
        return "Invalid", "index.html", "Please upload the correct soil image!"

    pred_name, output_page = classes.get(result, ("Unknown", "index.html"))

    # Return error if detected as Invalid class directly
    if pred_name == "Invalid":
        return "Invalid", "index.html", "Please upload the correct soil image!"

    return pred_name, output_page, None


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

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Execute Model Prediction with 75% confidence threshold check
        pred_name, output_page, error_msg = model_predict(file_path, SoilNet, threshold=0.75)

        # Handle Invalid / Non-soil image detection or low confidence
        if error_msg:
            if os.path.exists(file_path):
                os.remove(file_path)  # Delete invalid image file from static/uploads
            return render_template("index.html", error=error_msg)

        # Render corresponding soil HTML page on valid soil detection (>75% confidence)
        return render_template(output_page, image_path=filename)


if __name__ == "__main__":
    app.run(debug=True)