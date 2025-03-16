import os
import sys
import numpy as np
import random
import torch
from tqdm import tqdm as progress_bar
from model import Mamba2ScenarioModel
from utils import set_seed, setup_gpus, check_directories, plot_accs
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from torch import nn
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mamba_train(args, model, datasets, tokenizer):
    """Training function for the Mamba2 model."""
    # Set up loss function based on classification type
    if args.classification_type == 'multi_class':
        criterion = nn.CrossEntropyLoss()
    else:  # multi_label
        criterion = nn.BCEWithLogitsLoss()
    
    # Setup training dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    val_dataloader = get_dataloader(args, datasets['validation'], split='validation')
    
    # Prepare optimizer
    # Use parameter groups to apply different learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    
    # Setup scheduler
    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps * 0.1,  # 10% of training for warmup
        num_training_steps=total_steps
    )
    
    # Initialize mixed precision training if enabled
    scaler = GradScaler() if args.fp16 else None
    
    # Training loop
    train_accs = []
    eval_accs = []
    global_step = 0
    
    for epoch in range(args.n_epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{args.n_epochs} {'='*20}")
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            # Prepare inputs and move to device
            inputs, labels = prepare_inputs(batch)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if args.fp16:
                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss = loss / args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
            else:
                # Standard forward/backward pass
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
            
            # Calculate metrics
            if args.classification_type == 'multi_class':
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item()
            else:  # multi_label
                preds = (logits > 0.5).float()
                acc = (preds == labels).float().mean().item()
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_acc += acc
            
            # Print progress
            if global_step > 0 and global_step % 50 == 0:
                print(f"Step: {global_step}, Loss: {epoch_loss/(step+1):.4f}, Acc: {epoch_acc/(step+1):.4f}")
        
        # Epoch summary
        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_acc = epoch_acc / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f} | Training Acc: {avg_train_acc:.4f}")
        
        # Evaluation
        eval_acc, eval_loss = run_eval(args, model, datasets, tokenizer, split='validation')
        
        # Save metrics for plotting
        train_accs.append(avg_train_acc)
        eval_accs.append(eval_acc)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"mamba2_checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Plot learning curves
    plot_accs(train_accs, eval_accs, os.path.join(args.save_dir, "learning_curves"))
    
    return model

def run_eval(args, model, datasets, tokenizer, split='validation'):
    """Evaluation function for the Mamba2 model."""
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split=split)
    
    # Set up loss function based on classification type
    if args.classification_type == 'multi_class':
        criterion = nn.CrossEntropyLoss()
    else:  # multi_label
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
            inputs, labels = prepare_inputs(batch)
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            # Calculate metrics
            if args.classification_type == 'multi_class':
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item()
            else:  # multi_label
                preds = (logits > 0.5).float()
                acc = (preds == labels).float().mean().item()
            
            total_loss += loss.item()
            total_acc += acc
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    
    print(f'{split.capitalize()} Loss: {avg_loss:.4f} | {split.capitalize()} Acc: {avg_acc:.4f}')
    return avg_acc, avg_loss

def main():
    """Main function to set up and run the training/evaluation pipeline."""
    # Parse arguments
    args = params()
    
    # Setup environment
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)
    
    print(f"Using device: {device}")
    print(f"Number of GPUs: {args.n_gpu}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)
    
    # Check for cached features
    cache_results, already_exist = check_cache(args)
    
    if already_exist:
        print("Loading features from cache")
        features = cache_results
    else:
        print("Preparing new features")
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    
    # Process data into datasets
    datasets = process_data(args, features, tokenizer)
    
    # Determine the target size (number of classes)
    # This should match the number of emotion categories in GoEmotions
    target_size = 28  # GoEmotions has 28 emotions
    
    # Initialize model
    if args.model == 'mamba2':
        print(f"Initializing Mamba2 model from {args.mamba2_model_name}")
        model = Mamba2ScenarioModel(args, tokenizer, target_size).to(device)
        
        # Print model summary
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are trainable")
        
        # Run evaluation first if requested
        if args.do_eval:
            print("Running initial evaluation")
            run_eval(args, model, datasets, tokenizer, split='validation')
            run_eval(args, model, datasets, tokenizer, split='test')
        
        # Train model if requested
        if args.do_train:
            print("Starting training")
            model = mamba_train(args, model, datasets, tokenizer)
            
            # Run final evaluation
            print("Running final evaluation")
            run_eval(args, model, datasets, tokenizer, split='test')
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    print("All done!")

if __name__ == "__main__":
    main()