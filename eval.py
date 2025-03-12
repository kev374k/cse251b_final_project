import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # # EVAL FUNCTION # # #
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_correct_topk = 0
    totalPredCount = 0
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
            
            _, topk_predicted = torch.topk(logits, k=3, dim=1)
            for i in range(len(labels)):
                totalPredCount += 1
                if labels[i] in topk_predicted[i]:
                    total_correct_topk += 1

            total_correct += (predicted == labels).sum().item()
            total_labels.extend(labels.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')
    topk = total_correct_topk/totalPredCount

    return accuracy, precision, recall, f1, topk

# # # MAMBA EVAL FUNCTION # # #
def mamba_evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_correct_topk = 0
    totalPredCount = 0
    total_labels = []
    total_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['emotion'].to(device)

            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1)
            
            _, topk_predicted = torch.topk(outputs, k=3, dim=1)
            for i in range(len(labels)):
                totalPredCount += 1
                if labels[i] in topk_predicted[i]:
                    total_correct_topk += 1

            total_correct += (predicted == labels).sum().item()
            total_labels.extend(labels.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')
    topk = total_correct_topk/totalPredCount

    return accuracy, precision, recall, f1, topk