import argparse
import os

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="mamba2", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method")
    parser.add_argument("--mamba2_model_name", default="state-spaces/mamba-2.8b-hf", type=str,
                    help="Name or path of the pretrained Mamba2 model")
    # parser.add_argument("--mamba2_model_name", default="state-spaces/mamba-1.4b-hf", type=str,
    #                 help="Name or path of the pretrained Mamba2 model")
    parser.add_argument("--freeze_mamba", action="store_true",
                        help="Whether to freeze the Mamba2 parameters during training")
    parser.add_argument("--classification_type", default="multi_class", type=str,
                        choices=['multi_class', 'multi_label'], 
                        help="Type of classification problem")
    
    # Existing arguments
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='mamba2', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyperparameters
    parser.add_argument("--batch-size", default=4, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=5e-5, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=256, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.1, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=5, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=512, type=int,
                help="maximum sequence length to look back")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA fine-tuning")
    parser.add_argument('--loss_type', type=str, default='supcon', choices=['supcon', 'simclr'], 
                    help='Type of contrastive loss to use (supcon or simclr)')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--fp16", action="store_true",
                    help="Whether to use 16-bit (mixed) precision instead of 32-bit")

    args = parser.parse_args()
    return args