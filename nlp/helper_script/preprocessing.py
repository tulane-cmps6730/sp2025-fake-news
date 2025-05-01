import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm  # progress bar
from multiprocessing import cpu_count

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Register tqdm with pandas in case users still call .progress_apply()
tqdm.pandas()

# We will use spaCy's nlp.pipe to stream texts efficiently.
from tqdm import tqdm  # progress bar

def load_data(fake_path, true_path):
    """
    Load and combine fake and true news datasets
    """
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Add label column (0 for true, 1 for fake)
    fake_df['label'] = 1
    true_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    return df

def preprocess_text(text):
    """
    Preprocess text using spaCy and NLTK
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization using spaCy
    doc = nlp(text)
    
    # Lemmatization and stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

# Vectorised/parallel version using spaCy's pipe ---------------------------------

def preprocess_series(series, batch_size: int = 500, n_proc: int | None = None):
    """Preprocess an entire pandas Series using spaCy's pipe for speed.

    Args:
        series: The Series of strings to process.
        batch_size: How many texts per spaCy batch.
        n_proc: Number of processes. Defaults to (#cores - 1).

    Returns:
        pandas Series with processed strings (same index as *series*).
    """
    n_proc = n_proc or max(cpu_count() - 1, 1)
    stop_words = set(stopwords.words('english'))

    processed = []
    iterator = nlp.pipe(series.tolist(), batch_size=batch_size, n_process=n_proc,
                        disable=["parser", "ner"])  # disable heavy comps
    for doc in tqdm(iterator, total=len(series), desc="spaCy preprocessing"):
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
        processed.append(' '.join(tokens))
    return pd.Series(processed, index=series.index)

def prepare_dataset(df, test_size=0.2, random_state=42):
    """
    Prepare dataset by preprocessing text and splitting into train/test sets
    """
    # Preprocess title and text
    print("Preprocessing titles…")
    df['processed_title'] = preprocess_series(df['title'])
    print("Preprocessing main text…")
    df['processed_text'] = preprocess_series(df['text'])
    
    # Split into train and test sets
    X = df[['processed_title', 'processed_text']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test 