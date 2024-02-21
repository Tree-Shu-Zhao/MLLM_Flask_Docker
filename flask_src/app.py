import os

from flask import Flask, request, jsonify
from .model_controller import ModelController


app = Flask(__name__)

model_version = os.environ['MODEL_VERSION']
model = ModelController(model_version)

@app.route(f"/{model_version}")
def process():
    return jsonify(model.generate(image=request.files["image"], text=request.form["text"]))

if __name__ == "__main__":
    app.run(debug=True)

