from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_model")

text = "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()

print("FAKE" if prediction else "REAL")
