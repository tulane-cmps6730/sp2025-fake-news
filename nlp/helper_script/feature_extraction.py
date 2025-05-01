from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import os
import gc
from math import ceil
from tqdm import tqdm

# Load pre-trained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Use GPU if available (CUDA or Apple MPS) else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to tokenize and extract features from text
def extract_features(texts, batch_size=8, chunk_size=1000):
    all_embeddings = []
    
    # Process in chunks to save memory
    chunk_iter = range(0, len(texts), chunk_size)
    for chunk_start in tqdm(chunk_iter, total=ceil(len(texts)/chunk_size), desc="BERT feature extraction", unit="chunk"):
        chunk_texts = texts[chunk_start:chunk_start + chunk_size]
        chunk_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i+batch_size]
            
            # Ensure each element is a string and remove empty strings
            batch_texts = [str(text) for text in batch_texts if str(text).strip() != '']
            
            if not batch_texts:  # Skip if batch is empty
                continue
                
            # Tokenize input text with consistent padding
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding='max_length',  # Pad to max_length
                truncation=True,
                max_length=256
            )
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # Extract features using BERT
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get the embeddings from the last hidden state
            embeddings = outputs.last_hidden_state
            
            # Apply mean pooling to get fixed-size embeddings
            # This averages across the sequence length dimension
            pooled_embeddings = torch.mean(embeddings, dim=1)
            
            # Convert to float16 to reduce size and memory usage
            pooled_embeddings = pooled_embeddings.half()
            
            chunk_embeddings.append(pooled_embeddings.cpu())
            
            # Clear memory
            del inputs, outputs, embeddings, pooled_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate chunk embeddings
        if chunk_embeddings:
            chunk_embeddings = torch.cat(chunk_embeddings, dim=0)
            all_embeddings.append(chunk_embeddings)
            
            # Clear memory
            del chunk_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    return torch.cat(all_embeddings, dim=0)

# Function to save extracted features
def save_features(features, file_path):
    """
    Save features in float16 format to reduce storage space.
    This is suitable for our fake news classification task and
    follows common practices in production systems.
    """
    # Ensure features are in float16 format before saving
    features = features.half()
    torch.save(features, file_path)

# Main function
if __name__ == "__main__":
    # Resolve to <repo-root>/nlp/data/preprocessed regardless of CWD
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed'))
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Extract features for the entire training data
    print("Processing training data...")
    train_texts = train_data['processed_text'].tolist()
    train_features = extract_features(train_texts)
    save_features(train_features, os.path.join(data_dir, 'train_features.pt'))
    
    # Clear memory before processing test data
    del train_features, train_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Extract features for the entire test data
    print("Processing test data...")
    test_texts = test_data['processed_text'].tolist()
    test_features = extract_features(test_texts)
    save_features(test_features, os.path.join(data_dir, 'test_features.pt'))
    
    print("Features extracted and saved for training and test datasets.") 