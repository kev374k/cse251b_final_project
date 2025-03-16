import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # # TRAIN # # #
def train_and_eval(model, model_name, train_dataloader, val_dataloader, test_dataloader):
    best_accuracy = 0
    best_f1 = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
        
        if model_name == "QLoraSCHED":
            scheduler.step()

        model.eval()
        
        accuracy, precision, recall, f1, topk = evaluate(model, val_dataloader, device)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Right Answer in Top 3: {topk:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{model_name}_best_accuracy_model.pth')
            print(f"Saved best accuracy model with accuracy: {best_accuracy:.4f}")
            
    accuracy, precision, recall, f1, topk = evaluate(model, test_dataloader, device)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Right Answer in Top 3: {topk:.4f}")


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