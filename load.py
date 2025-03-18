import pandas as pd
from transformers import BertTokenizer, AutoTokenizer

def load_data():
    """
    Load the GoEmotions dataset from tsv files.
    Returns a dictionary with train, validation, and test splits.
    """
    # Load the GoEmotions dataset
    train_data = pd.read_csv('data/train.tsv', sep='\t', header=None, names=['text', 'labels', 'id'])
    val_data = pd.read_csv('data/dev.tsv', sep='\t', header=None, names=['text', 'labels', 'id'])
    test_data = pd.read_csv('data/test.tsv', sep='\t', header=None, names=['text', 'labels', 'id'])

    # Convert labels to list of integers
    train_data['labels'] = train_data['labels'].apply(lambda x: [int(i) for i in x.split(',')])
    val_data['labels'] = val_data['labels'].apply(lambda x: [int(i) for i in x.split(',')])
    test_data['labels'] = test_data['labels'].apply(lambda x: [int(i) for i in x.split(',')])

    # Convert to dictionary format and rename 'labels' to 'label'
    dataset = {
        'train': [{'text': row['text'], 'label': row['labels'][0], 'id': row['id']} for _, row in train_data.iterrows()],
        'validation': [{'text': row['text'], 'label': row['labels'][0], 'id': row['id']} for _, row in val_data.iterrows()],
        'test': [{'text': row['text'], 'label': row['labels'][0], 'id': row['id']} for _, row in test_data.iterrows()],
    }
    
    print(f"Loaded {len(dataset['train'])} training examples")
    print(f"Loaded {len(dataset['validation'])} validation examples")
    print(f"Loaded {len(dataset['test'])} test examples")
    
    return dataset

def load_tokenizer(args):
    """
    Load the appropriate tokenizer based on the selected model.
    """
    if args.model == 'mamba2':
        # Use the correct tokenizer for Mamba2
        # Mamba-2.8b typically uses the EleutherAI/gpt-neox-20b tokenizer
        tokenizer_name = "AntonV/mamba2-130m-hf"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure we have padding token set correctly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Set padding to left (important for autoregressive models like Mamba)
        tokenizer.padding_side = "left"
        
        print(f"Loaded tokenizer for Mamba2: {tokenizer_name}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Padding token: {tokenizer.pad_token}")
        
    else:
        # Default to BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
        print("Loaded BERT tokenizer")
        
    return tokenizer