import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm as progress_bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args, dataset, split='train'):
    """
    Create a DataLoader with appropriate sampler for the given dataset split.
    
    Args:
        args: Command line arguments
        dataset: Dataset object
        split: Data split ('train', 'validation', or 'test')
    
    Returns:
        DataLoader object
    """
    # Use RandomSampler for training to shuffle data, SequentialSampler for eval
    sampler = RandomSampler(dataset) if split == 'train' else SequentialSampler(dataset)
    
    # Get collate function from dataset
    collate = dataset.collate_func
    
    # Create dataloader with appropriate batch size
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.batch_size, 
        collate_fn=collate
    )
    
    print(f"Created {split} dataloader with {len(dataloader)} batches (batch size: {args.batch_size})")
    return dataloader

def prepare_inputs(batch, use_text=False):
    """
    Prepare inputs for the model. Compatible with Mamba2 models.
    
    Args:
        batch: Batch from dataloader
        use_text: Whether to return text content as well
    
    Returns:
        inputs: Dictionary with input_ids and attention_mask
        targets: Target labels
        (optional) target_text: Text representation of targets
    """
    # Move tensors to device
    input_ids = batch[0].to(device)
    attention_mask = batch[2].to(device)
    labels = batch[3].to(device)
    
    # Create inputs dictionary (Mamba2 only needs input_ids and attention_mask)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    # Return with or without text
    if use_text:
        target_text = batch[4]
        return inputs, labels, target_text
    else:
        return inputs, labels

def check_cache(args):
    """
    Check if cached features exist.
    
    Args:
        args: Command line arguments
    
    Returns:
        Tuple of (cache_path_or_data, already_exists)
    """
    # Define cache path
    folder = 'cache'
    os.makedirs(os.path.join(args.input_dir, folder), exist_ok=True)
    cache_path = os.path.join(args.input_dir, folder, f'{args.dataset}.pkl')
    
    # Determine whether to use cache
    use_cache = not args.ignore_cache

    # Return cache path or data
    if os.path.exists(cache_path) and use_cache:
        print(f'Loading features from cache at {cache_path}')
        with open(cache_path, 'rb') as f:
            results = pkl.load(f)
        return results, True
    else:
        print(f'Cache not found or ignored. Will create new features...')
        return cache_path, False

def prepare_features(args, data, tokenizer, cache_path):
    """
    Prepare features for model input from raw data.
    
    Args:
        args: Command line arguments
        data: Raw data dictionary with train/validation/test splits
        tokenizer: Tokenizer for the model
        cache_path: Path to save the cached features
    
    Returns:
        Dictionary of features for each split
    """
    all_features = {}
    LABELS = {}  # To track label mappings

    for split, examples in data.items():
        print(f"Preparing features for {split} split...")
        feats = []
        
        # Process each example in the split
        for example in progress_bar(examples, total=len(examples)):
            # Tokenize text with appropriate padding for Mamba2 (left padding)
            embed_data = tokenizer(
                example['text'], 
                padding='max_length', 
                truncation=True, 
                max_length=args.max_len, 
                return_tensors='pt'
            )
            
            # Convert tensors to lists for storage
            input_ids = embed_data['input_ids'].squeeze(0).tolist()
            attention_mask = embed_data['attention_mask'].squeeze(0).tolist()
            
            # Process label
            label = example['label']
            if label not in LABELS:
                LABELS[label] = len(LABELS)
            label_id = LABELS[label]
            
            # Create instance with processed data
            instance = MambaInstance(
                {'input_ids': input_ids, 'attention_mask': attention_mask}, 
                {'text': example['text'], 'label': label_id}
            )
            feats.append(instance)
        
        # Store features for this split
        all_features[split] = feats
        print(f'Created {len(feats)} features for {split} split')

    # Cache features for future use
    print(f'Saving features to cache at {cache_path}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pkl.dump(all_features, f)
    
    return all_features

class MambaInstance:
    """
    Container class for a single example's processed features.
    """
    def __init__(self, embed_data, example):
        # Store embeddings for model input
        self.embedding = embed_data['input_ids']
        self.input_mask = embed_data['attention_mask']
        
        # Store labels and text
        self.scenario_label = example['label']
        self.text = example['text']
        self.label_text = str(example['label'])

def process_data(args, features, tokenizer):
    """
    Process feature data into Dataset objects.
    
    Args:
        args: Command line arguments
        features: Dictionary of feature data for each split
        tokenizer: Tokenizer for the model
    
    Returns:
        Dictionary of Dataset objects for each split
    """
    datasets = {}
    
    # Create dataset for each split
    for split, feat in features.items():
        datasets[split] = MambaDataset(feat, tokenizer, split)
        print(f"Created {split} dataset with {len(datasets[split])} examples")
    
    return datasets

class MambaDataset(Dataset):
    """
    Dataset class for Mamba model.
    """
    def __init__(self, data, tokenizer, split='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_func(self, batch):
        """
        Collate function to create batches from instances.
        For Mamba2, we don't need token_type_ids (segment_ids).
        """
        # Collect inputs and labels
        input_ids = torch.tensor([f.embedding for f in batch], dtype=torch.long)
        input_masks = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.scenario_label for f in batch], dtype=torch.long)
        
        # Also collect text labels for reference
        label_texts = [f.label_text for f in batch]
        
        # Return batch (None for token_type_ids which Mamba doesn't use)
        return input_ids, None, input_masks, label_ids, label_texts