from flask import Flask, request, jsonify
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model = BERT_Arch(bert)

path = 'model/best_model.pt'
model.load_state_dict(torch.load(path))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print("Received data:", data)
    text = data.get("text", "")
    
    if not isinstance(text, str) or text == "":
        return jsonify({"error": "Input should be a non-empty string."}), 400

    #tokenize and encode the input text
    MAX_LENGTH = 15
    tokens = tokenizer.encode_plus(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        preds = model(input_ids, attention_mask)
        preds = preds.detach().cpu().numpy()

    prediction = np.argmax(preds, axis=1).item()
    is_fake = prediction == 1
    return jsonify({"is_fake": is_fake})

if __name__ == "__main__":
    app.run(port=5000, debug=True, threaded=True)
