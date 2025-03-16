from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn
from dataloader import load_train, load_val, load_test
from train import train_and_eval

#LORA
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRALinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear.weight.requires_grad = False
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.alpha = alpha

    def forward(self, x):
        frozen_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(x))
        lora_out = lora_out * self.alpha
        return frozen_out + lora_out

#QLORA
class QLoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, quantize_base=False):
        super(QLoRALinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        if quantize_base:
            def to_nf4(weight):
                return weight
            self.linear.weight = to_nf4(self.linear.weight)
        self.linear.weight.requires_grad = False
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.alpha = alpha

    def forward(self, x):
        frozen_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(x))
        lora_out = lora_out * self.alpha
        return frozen_out + lora_out
    

class BertWithLoRA(BertForSequenceClassification):
    def __init__(self, config):
        super(BertWithLoRA, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LoRALinear(config.hidden_size, config.num_labels, rank=8, alpha=1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.config.num_labels), labels.view(-1))
        return loss, outputs
class BertWithQLoRA(BertForSequenceClassification):
    def __init__(self, config):
        super(BertWithQLoRA, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = QLoRALinear(config.hidden_size, config.num_labels, rank=8, alpha=1.0, quantize_base=True)

class BertWithQLoRAAdjusted(BertForSequenceClassification):
    def __init__(self, config):
        super(BertWithQLoRAAdjusted, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = QLoRALinear(config.hidden_size, config.num_labels, rank=15, alpha=1.0, quantize_base=True)


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def bert_tokenize_function(examples):
    return bert_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)


train_dataloader = load_train(bert_tokenize_function)
val_dataloader = load_val(bert_tokenize_function)
test_dataloader = load_test(bert_tokenize_function)

# NONE
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28) 
# train_and_eval(model, 'bert', train_dataloader, val_dataloader, test_dataloader)

# LORA
model = BertWithLoRA.from_pretrained('bert-base-uncased', num_labels=28)
train_and_eval(model, 'Lora', train_dataloader, val_dataloader, test_dataloader)

# QLoRA
# model = BertWithQLoRA.from_pretrained('bert-base-uncased', num_labels=28)
# train_and_eval(model, 'QLora', train_dataloader, val_dataloader, test_dataloader)

# QLoRA rank 15
# model = BertWithQLoRAAdjusted.from_pretrained('bert-base-uncased', num_labels=28)
# train_and_eval(model, 'QLoraADJ', train_dataloader, val_dataloader, test_dataloader)

# QLoRA and Scheduler
# model = BertWithQLoRAAdjusted.from_pretrained('bert-base-uncased', num_labels=28)
# train_and_eval(model, 'QLoraSCHED', train_dataloader, val_dataloader, test_dataloader)