import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model


model_path = "state-spaces/mamba-130m-hf" # Mamba 1
#model_path = "AntonV/mamba2-130m-hf" # Mamba 2

class MambaModel(nn.Module):
  def __init__(self, tokenizer, embed_dim=768, hidden_dim=768*2, drop_rate=.1, target_size=28):
    super().__init__()
    self.tokenizer = tokenizer
    self.target_size = target_size
    self.dropout = nn.Dropout(drop_rate)
    self.classify = Classifier(embed_dim, hidden_dim, target_size)
    self.encoder = AutoModelForCausalLM.from_pretrained(model_path, use_mambapy=True)
      
    '''
    # Get config for quantized model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, # Load model in 4bit
        bnb_4bit_compute_dtype= torch.bfloat16, # Compute in bfloat16
        bnb_4bit_use_double_quant=True,
    )
    # Load model with Quantization config
    self.encoder = AutoModelForCausalLM.from_pretrained(model_path,
                                                        quantization_config=quant_config,
                                                        use_mambapy=True,
                                                        device_map="auto")
    
    # Apply LoRA to self.encoder
    peft_config = LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    self.encoder = get_peft_model(self.encoder, peft_config)
    peft_config = LoraConfig( 
        inference_mode=False, 
        r=15, 
        lora_alpha=32, 
        lora_dropout=0.1, 
        target_modules='all-linear'
    )
    '''
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))
    
  def forward(self, inputs): #, targets
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    #with torch.autocast("cuda"):
    outputs = self.encoder(inputs, output_hidden_states=True)
    cls_token = outputs.hidden_states[-1][:,-1,:]
    dropped_out = self.dropout(cls_token)
    logits = self.classify(dropped_out)
    return logits
  
class Classifier(nn.Module):
  def __init__(self, embed_dim, hidden_dim, target_size):
    super().__init__()
    input_dim = embed_dim
    self.top = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit
