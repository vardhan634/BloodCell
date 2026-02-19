import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load model
model = load_model("Blood_cell.h5")

class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = class_labels[class_index]

    return label

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            prediction = predict_image(filepath)

            return render_template("result.html",
                                   prediction=prediction,
                                   image_path=filepath)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)