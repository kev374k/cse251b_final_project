import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Mamba2ScenarioModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.target_size = target_size
        
        # Load pretrained Mamba2 model
        self.config = AutoConfig.from_pretrained(args.mamba2_model_name)
        self.mamba = AutoModel.from_pretrained(args.mamba2_model_name)
        
        # Freeze the model parameters if needed
        if args.freeze_mamba:
            for param in self.mamba.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, target_size)
    
    def forward(self, inputs):
        # Get the input_ids and attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Pass through the pretrained Mamba2
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the hidden states from the last layer
        hidden_states = outputs.last_hidden_state
        
        # Use the [CLS] token representation or the mean of all tokens
        # For classification tasks, often the first token ([CLS]) is used
        sentence_repr = hidden_states[:, 0, :]  # Use the first token
        
        dropped_out = self.dropout(sentence_repr)
        logits = self.classify(dropped_out)
        return logits

class Classifier(nn.Module):
    def __init__(self, args, target_size):
        super().__init__()
        # Adjust input dimension to match Mamba2's hidden size
        input_dim = self.get_input_dim(args)
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)
        
        # For multi-class classification
        if args.classification_type == 'multi_class':
            self.activation = nn.Softmax(dim=-1)
        else:  # For multi-label classification
            self.activation = nn.Sigmoid()

    def get_input_dim(self, args):
        if args.model == 'mamba2':
            # Use the actual hidden size from the pretrained model
            return 2048  # Mamba-2.8b hidden size
        else:
            return args.embed_dim
            
    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.activation(self.bottom(middle))
        return logit