from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import tqdm
from dataloader import load_train, load_val, load_test
from eval import evaluate


roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def roberta_tokenize_function(examples):
    return roberta_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)

# # # MAIN # # #
def main():
    best_accuracy = 0
    best_f1 = 0

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=28)

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
        
        accuracy, precision, recall, f1, topk = evaluate(model, val_dataloader, device)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Right Answer in Top 3: {topk:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'roberta_best_accuracy_model.pth')
            print(f"Saved best accuracy model with accuracy: {best_accuracy:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'roberta_best_f1_model.pth')
            print(f"Saved best F1 model with F1 score: {best_f1:.4f}")
            
    accuracy, precision, recall, f1, topk = evaluate(model, test_dataloader, device)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Right Answer in Top 3: {topk:.4f}")


train_dataloader = load_train(roberta_tokenize_function)
val_dataloader = load_val(roberta_tokenize_function)
test_dataloader = load_test(roberta_tokenize_function)
main()