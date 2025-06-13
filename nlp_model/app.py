from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_model")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    is_fake = prediction == 1
    print(f"Classified as {'FAKE' if is_fake else 'REAL'} --> {text}")
    return jsonify({"is_fake": is_fake})

if __name__ == "__main__":
    app.run(port=5000)
