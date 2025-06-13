from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

df = pd.read_csv("data/merged_data.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df['title'], df['label'], test_size=0.2)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = FakeNewsDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = FakeNewsDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="models/bert_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model("models/bert_model")
tokenizer.save_pretrained("models/bert_model")
