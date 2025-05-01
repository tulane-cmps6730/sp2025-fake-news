import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import os
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Word2VecFeatureExtractor:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        
    def _prepare_sentences(self, texts: List[str]) -> List[List[str]]:
        # Handle NaN values by converting them to empty strings
        texts = [str(text) if pd.notna(text) else "" for text in texts]
        return [text.split() for text in texts]
    
    def train_word2vec(self, texts: List[str]) -> None:
        logger.info("Preparing sentences for Word2Vec training...")
        sentences = self._prepare_sentences(texts)
        
        logger.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        if self.model is None:
            raise ValueError("Word2Vec model has not been trained yet")
            
        # Handle NaN values
        if pd.isna(text):
            return np.zeros(self.vector_size)
            
        words = str(text).split()
        word_vectors = []
        
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
                
        if not word_vectors:
            return np.zeros(self.vector_size)
            
        return np.mean(word_vectors, axis=0)
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Processing dataset...")
        
        # Combine title and text for training Word2Vec
        all_texts = pd.concat([df['processed_title'], df['processed_text']]).tolist()
        self.train_word2vec(all_texts)
        
        # Get embeddings for titles and texts
        title_embeddings = np.array([self.get_text_embedding(text) for text in df['processed_title']])
        text_embeddings = np.array([self.get_text_embedding(text) for text in df['processed_text']])
        
        # Combine title and text embeddings
        X = np.hstack([title_embeddings, text_embeddings])
        y = df['label'].values
        
        return X, y

def main():
    # Load data
    # Resolve to <repo-root>/nlp/data/preprocessed regardless of CWD
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed'))
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    logger.info("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Initialize feature extractor
    feature_extractor = Word2VecFeatureExtractor()
    
    # Process datasets
    logger.info("Processing training data...")
    X_train, y_train = feature_extractor.process_dataset(train_df)
    
    logger.info("Processing test data...")
    X_test, y_test = feature_extractor.process_dataset(test_df)
    
    # Save processed data
    output_dir = os.path.join('data', 'word2vec_features')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    logger.info("Feature extraction completed and saved to data/word2vec_features/")

if __name__ == "__main__":
    main() 