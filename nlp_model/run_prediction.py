from flask import Flask
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
      self.relu =  nn.ReLU()
      self.fc1 = nn.Linear(768,512)
      self.fc2 = nn.Linear(512,2)
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

unseen_news_text = [
    "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing",  # Fake
    "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",  # Fake
    "Obama funds alien research in Area 51",  # Fake
    "Clinton behind secret child trafficking ring",  # Fake
    "Pizzagate scandal resurfaces on social media",  # Fake
    "White House denies moon landing conspiracy",  # Fake
    "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",  # True
    "Trump administration issues new rules on U.S. visa waivers",  # True
    "Senate passes immigration reform bill",  # True
    "Biden signs executive order on AI safety",  # True
    "UN report on climate change sparks debate",  # True
    "CDC warns of new flu outbreak this winter"  # True
]

MAX_LENGTH = 15
tokens_unseen = tokenizer.batch_encode_plus(
    unseen_news_text,
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

input_ids = tokens_unseen['input_ids']
attention_mask = tokens_unseen['attention_mask']

with torch.no_grad():
    preds = model(input_ids, attention_mask)
    preds = preds.detach().cpu().numpy()

predictions = np.argmax(preds, axis=1)

for i, prediction in enumerate(predictions):
    result = "FAKE" if prediction == 1 else "REAL"
    print(f"{unseen_news_text[i]} -> {result}")
