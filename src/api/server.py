import json
import os

import joblib
import validators
from flask import Flask, jsonify, redirect, request

app = Flask(__name__)


def load_classifier():
    return joblib.load(os.path.join("resources", "models", "classifier.pkl"))


classifier = load_classifier()


@app.route("/api/v1.0/classify/", methods=["POST"])
def classify():
    content = request.json

    if not content or "links" not in content:
        return jsonify({"error": "Invalid input: not found links"}), 400

    if not all([validators.url(x["link"]) for x in content["links"]]):
        return jsonify({"error": "Invalid input: invalid link format"}), 400

    json_data = content["links"]
    links = []
    for entity in json_data:
        links.append(entity["link"])

    predicted = classifier.predict(json_data)

    out = {"links": [{i[0]: bool(i[1])} for i in zip(links, predicted)]}
    return json.dumps(out)


@app.route("/status")
def status():
    return "OK"


@app.route("/")
def to_digest():
    return redirect("https://pythondigest.ru/")


if __name__ == "__main__":
    app.run(debug=True)
