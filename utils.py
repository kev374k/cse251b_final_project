import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def check_directories(args):
    """
    Create necessary directories for saving outputs.
    
    Args:
        args: Command line arguments
    
    Returns:
        Updated args with save directory path
    """
    # Create main output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Create task-specific directory
    folder = args.task
    save_path = os.path.join(args.output_dir, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created task directory: {save_path}")
    
    # Add timestamp to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(save_path, f"run_{timestamp}")
    os.makedirs(run_path)
    print(f"Created run directory: {run_path}")
    
    # Set save directory in args
    args.save_dir = run_path
    
    # Create cache directory if it doesn't exist
    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print(f"Created cache directory: {cache_path}")
    
    # Setup logging
    setup_logging(args)
    
    return args

def setup_logging(args):
    """
    Set up logging configuration.
    
    Args:
        args: Command line arguments
    """
    log_file = os.path.join(args.save_dir, 'run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Arguments: {args}")

def set_seed(args):
    """
    Set random seeds for reproducibility.
    
    Args:
        args: Command line arguments with seed value
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    # Set deterministic behavior for reproducibility
    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    logging.info(f"Set random seed to {args.seed}")

def setup_gpus(args):
    """
    Set up GPU environment.
    
    Args:
        args: Command line arguments
    
    Returns:
        Updated args with GPU information
    """
    n_gpu = 0  # default to 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        # Log GPU information
        for i in range(n_gpu):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    args.n_gpu = n_gpu
    logging.info(f"Using {n_gpu} GPUs")
    
    # Set device
    if n_gpu > 0:
        torch.cuda.set_device(0)  # Use the first GPU by default
    
    return args

def plot_accs(train_accs, val_accs, filename):
    """
    Plot training and validation accuracy curves and save the figure.
    
    Args:
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        filename: Output filename (without extension)
    """
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()
    
    # Also save data as CSV
    with open(f"{filename}.csv", 'w') as f:
        f.write("epoch,train_acc,val_acc\n")
        for i, (t_acc, v_acc) in enumerate(zip(train_accs, val_accs)):
            f.write(f"{i+1},{t_acc},{v_acc}\n")
    
    logging.info(f"Saved learning curves to {filename}.png and {filename}.csv")

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        String with memory usage information
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    else:
        return "No GPU available"