import os

from flask import Flask, request, jsonify
from models import ModelController
from PIL import Image


app = Flask(__name__)

model_version = os.environ['MODEL_VERSION']
model = ModelController(model_version)


@app.route(f"/{model_version}", methods=["POST"])
def process():
    text = request.form["text"]
    image = Image.open(request.files["image"]).convert("RGB")
    return jsonify(model.generate(text=text, image=image))

if __name__ == "__main__":
    app.run(debug=True)

