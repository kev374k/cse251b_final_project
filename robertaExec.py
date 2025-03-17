from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, RobertaModel
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
    

class RobertaWithLoRA(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaWithLoRA, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LoRALinear(config.hidden_size, config.num_labels, rank=8, alpha=1.0)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        last_hidden_state = outputs.last_hidden_state
        cls_token_output = last_hidden_state[:, 0, :]  # Take the output of the CLS token
        
        cls_token_output = self.dropout(cls_token_output)
        outputs = self.classifier(cls_token_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, self.config.num_labels), labels.view(-1))
        return loss, outputs
    
class RobertaWithQLoRA(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaWithQLoRA, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = QLoRALinear(config.hidden_size, config.num_labels, rank=8, alpha=1.0, quantize_base=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take the output of the CLS token
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Ensure labels are not reshaped incorrectly
            loss_fct = nn.CrossEntropyLoss()
            # Check if labels need reshaping to match batch size
            if labels.shape[0] != outputs.shape[0]:
                print("Batch size mismatch detected. Labels shape:", labels.shape, "Outputs shape:", outputs.shape)
                # Handle the mismatch, e.g., by slicing labels to match outputs
                labels = labels[:outputs.shape[0]]
            loss = loss_fct(outputs, labels)
        
        return loss, outputs

class RobertaWithQLoRAAdjusted(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaWithQLoRAAdjusted, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = QLoRALinear(config.hidden_size, config.num_labels, rank=15, alpha=1.0, quantize_base=True)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take the output of the CLS token
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Ensure labels are not reshaped incorrectly
            loss_fct = nn.CrossEntropyLoss()
            # Check if labels need reshaping to match batch size
            if labels.shape[0] != outputs.shape[0]:
                print("Batch size mismatch detected. Labels shape:", labels.shape, "Outputs shape:", outputs.shape)
                # Handle the mismatch, e.g., by slicing labels to match outputs
                labels = labels[:outputs.shape[0]]
            loss = loss_fct(outputs, labels)
        
        return loss, outputs


roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def roberta_tokenize_function(examples):
    return roberta_tokenizer(examples['text'], padding='max_length', truncation=True, max_length=80)


train_dataloader = load_train(roberta_tokenize_function)
val_dataloader = load_val(roberta_tokenize_function)
test_dataloader = load_test(roberta_tokenize_function)

# NONE
# model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=28)
# train_and_eval(model, 'roberta', train_dataloader, val_dataloader, test_dataloader)

# LORA
model = RobertaWithLoRA.from_pretrained('roberta-large', num_labels=28)
train_and_eval(model, 'Lora', train_dataloader, val_dataloader, test_dataloader)

# QLoRA
# model = RobertaWithQLoRA.from_pretrained('roberta-large', num_labels=28)
# train_and_eval(model, 'QLora', train_dataloader, val_dataloader, test_dataloader)

# QLoRA rank 15
# model = RobertaWithQLoRAAdjusted.from_pretrained('roberta-large', num_labels=28)
# train_and_eval(model, 'QLoraADJ', train_dataloader, val_dataloader, test_dataloader)

# QLoRA and Scheduler
# model = RobertaWithQLoRAAdjusted.from_pretrained('roberta-large', num_labels=28)
# train_and_eval(model, 'QLoraSCHED', train_dataloader, val_dataloader, test_dataloader)