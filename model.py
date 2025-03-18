import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class Mamba2ScenarioModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.hidden_size = 768  # AntonV/mamba2-130m-hf has 768 hidden size
        
        # Load pretrained Mamba2 model with optimized loading
        print(f"Loading {args.mamba2_model_name}...")
        try:
            self.config = AutoConfig.from_pretrained(args.mamba2_model_name)
            self.mamba = AutoModelForCausalLM.from_pretrained(
                args.mamba2_model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if args.fp16 else torch.float32
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU loading...")
            self.mamba = AutoModelForCausalLM.from_pretrained(
                args.mamba2_model_name,
                trust_remote_code=True,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        # Freeze parameters if specified
        if args.freeze_mamba:
            print("Freezing Mamba2 parameters")
            for param in self.mamba.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, target_size, self.hidden_size)
    
    def forward(self, inputs):
        # Extract inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Forward pass with memory optimization
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.mamba(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get the last hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Get the last token representation for classification
        # For left-padded sequences, we need to use attention_mask to find the last token
        batch_size = input_ids.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 because 0-indexed
        
        # Create indices to gather the last token for each sequence
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        last_token_indices = torch.stack([batch_indices, sequence_lengths], dim=1)
        
        # Get sentence representation from the last token
        sentence_repr = hidden_states[batch_indices, sequence_lengths]
        
        # Apply dropout and classification
        dropped_out = self.dropout(sentence_repr)
        logits = self.classify(dropped_out)
        return logits

class Classifier(nn.Module):
    def __init__(self, args, target_size, hidden_size):
        super().__init__()
        # Use the provided hidden size from the model
        self.top = nn.Linear(hidden_size, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, target_size)
        
        # For multi-class classification
        if args.classification_type == 'multi_class':
            self.activation = nn.Identity()  # We'll use CrossEntropyLoss which applies softmax internally
        else:  # For multi-label classification
            self.activation = nn.Identity()  # We'll use BCEWithLogitsLoss which applies sigmoid internally
            
    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logits = self.bottom(middle)  # Raw logits
        return logits  # Return raw logits, let loss functions handle the activation