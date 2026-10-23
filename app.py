
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    return jsonify({
        "message": "Training is not triggered by this endpoint. "
                   "Run the training pipeline separately if needed."
    }), 200


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"error": "Please provide an image in the request."}), 400

        decodeImage(data["image"], clApp.filename)

        result = clApp.classifier.predict()
        return jsonify(result), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server logs."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
