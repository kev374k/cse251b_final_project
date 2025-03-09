from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


# # # LOAD TRAIN # # #
df = pd.read_csv('data/train.tsv', sep='\t', header=None)
df = df.drop(df.columns[2], axis=1)

newrows = [] # ONLY HANDLES ROWS WITH 1 EMOTION. COMMENTED CODE ADDS NEW TRAINING ROWS FOR MULTIPLE EMOTIONS
for i in range(0, df.shape[0]): 
    if "," in df[1].iloc[i]:
        # currEmotion = df[1].iloc[i].split(",")
        # currText = df[0].iloc[i]
        # for j in currEmotion:
        #     newrows.append([currText, j])
        pass
    else:
        newrows.append([df[0].iloc[i], int(df[1].iloc[i])])

df = pd.DataFrame(newrows)
        
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': df[0], 'emotion': df[1]}))
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'emotion'])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)


# # # LOAD VALIDATION # # #
df = pd.read_csv('data/dev.tsv', sep='\t', header=None)
df = df.drop(df.columns[2], axis=1)

newrows = [] # ONLY HANDLES ROWS WITH 1 EMOTION. COMMENTED CODE ADDS NEW VALIDATION ROWS FOR MULTIPLE EMOTIONS
for i in range(0, df.shape[0]): # NEED FIND WAY TO HANDLE IF WE DECIDE TO PURSUE. MULTIPLE EMOTIONS COULD CONFUSE THE MODEL IN THIS FORMAT (LIKE THE TESLA CASE PROF MENTIONED)
    if "," in df[1].iloc[i]:
        # currEmotion = df[1].iloc[i].split(",")
        # currText = df[0].iloc[i]
        # for j in currEmotion:
        #     newrows.append([currText, j])
        pass
    else:
        newrows.append([df[0].iloc[i], int(df[1].iloc[i])])

df = pd.DataFrame(newrows)
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': df[0], 'emotion': df[1]}))
val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'emotion'])
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=8)



# # # EVAL FUNCTION # # #
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_labels = []
    total_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['emotion'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_labels.extend(labels.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')

    return accuracy, precision, recall, f1


# # # MAIN # # #
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(3):
    model.train()
    total_loss = 0
    with tqdm.tqdm(train_dataloader, unit="batch", desc=f"Epoch {epoch+1}") as pbar:
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['emotion'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_dataloader)}')

    model.eval()
    
    accuracy, precision, recall, f1 = evaluate(model, val_dataloader, device)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")