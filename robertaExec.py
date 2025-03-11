from transformers import RobertaForSequenceClassification, RobertaTokenizer
from dataloader import load_train, load_val, load_test
from train import train_and_eval


roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def roberta_tokenize_function(examples):
    return roberta_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)


train_dataloader = load_train(roberta_tokenize_function)
val_dataloader = load_val(roberta_tokenize_function)
test_dataloader = load_test(roberta_tokenize_function)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=28)
train_and_eval(model, 'roberta', train_dataloader, val_dataloader, test_dataloader)