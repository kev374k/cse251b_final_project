from transformers import BertForSequenceClassification, BertTokenizer
from dataloader import load_train, load_val, load_test
from train import train_and_eval


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_tokenize_function(examples):
    return bert_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)


train_dataloader = load_train(bert_tokenize_function)
val_dataloader = load_val(bert_tokenize_function)
test_dataloader = load_test(bert_tokenize_function)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28) 
train_and_eval(model, 'bert', train_dataloader, val_dataloader, test_dataloader)