import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


class IMDBDataset(Dataset):
    def __init__(self, csv_file, device, model_name_or_path="bert-base-uncased", max_length=512):
        self.device = device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.polarity.unique()
        labels_dict = {label: idx for idx, label in enumerate(self.labels)}
        self.df["polarity"] = self.df["polarity"].map(labels_dict)
        
        # Vérifiez les labels après mapping
        if self.df["polarity"].max() >= len(self.labels) or self.df["polarity"].min() < 0:
            raise ValueError("Les labels sont hors de portée après le mapping. Vérifiez le DataFrame d'entrée.")
        
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        review_text = str(self.df.review[index])
        label_review = self.df.polarity[index]
        
        inputs = self.tokenizer(
            review_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = torch.tensor(label_review)

        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device),
        }

class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

def training_step(model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        
        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels.long())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader.dataset)

def evaluation(model, data_loader, loss_fn):
    model.eval()
    correct_predictions = 0
    losses = []
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, pred = output.max(dim=1)
            correct_predictions += torch.sum(pred == labels)
            loss = loss_fn(output, labels.long())
            losses.append(loss.item())
            
    return np.mean(losses), correct_predictions / len(data_loader.dataset)

def preprocess_csv(file_path):
    df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
    
    data = df['review'].astype(str)
    
    processed = data.str.replace(r"#(\w+)", "", regex=True)
    processed = processed.str.lower()
    
    df['review'] = processed
    return df

def main():
    print("Training ....")
    N_EPOCHS = 1
    LR = 2e-5
    BATCH_SIZE = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = preprocess_csv("/kaggle/input/allocine-french-movie-reviews/train.csv")
    test = preprocess_csv("/kaggle/input/allocine-french-movie-reviews/test.csv")
    valid = preprocess_csv("/kaggle/input/allocine-french-movie-reviews/valid.csv")

    train.to_csv("train_IMDB.csv", index=False)
    test.to_csv("test_IMDB.csv", index=False)
    valid.to_csv("valid_IMDB.csv", index=False)

    train_dataset = IMDBDataset(csv_file="train_IMDB.csv", device=device, max_length=100)
    test_dataset = IMDBDataset(csv_file="test_IMDB.csv", device=device, max_length=100)
    valid_dataset = IMDBDataset(csv_file="valid_IMDB.csv", device=device, max_length=100)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE)

    model = CustomBert()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_valid, accuracy_valid = evaluation(model, valid_dataloader, loss_fn)
        loss_test, accuracy_test = evaluation(model, test_dataloader, loss_fn)

        print(f"Train loss: {loss_train} | Valid loss: {loss_valid} | Valid Accuracy: {accuracy_valid} | Test loss: {loss_test} | Test Accuracy: {accuracy_test}")
         
    torch.save(model.state_dict(), "my_custom_bert3.pth")

if __name__ == "__main__":
    main()
